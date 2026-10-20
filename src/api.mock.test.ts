import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clampSettings,
  copiedPatch,
  createMockApi,
  defaultSettings,
  expandNameTemplate,
  libraryOf,
  lookSlug,
  MOCK_APPLE_ANSWER,
  MOCK_APPLE_INFO,
  MOCK_DEFAULT_INSTRUCTIONS,
  MOCK_ESTIMATES,
  mockEstimates,
  optionsFromLocation,
  percentSetting,
  resolveSettings,
  type MockApi,
} from "./api.mock";
import type { AppError, ExportJob, ExportOptions, ExportProgress, PhotoProgress, ThumbnailReady } from "./api/types";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW"];

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

/** What a rejected command carries. */
async function rejection(promise: Promise<unknown>): Promise<AppError> {
  try {
    await promise;
  } catch (error) {
    return error as AppError;
  }
  throw new Error("expected the command to reject");
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

describe("session", () => {
  it("starts empty", async () => {
    expect(await api.sessionState()).toEqual({
      folder_name: null,
      photos: [],
      current_index: null,
      picked: [],
      default_export_folder: null,
    });
  });

  it("opens a folder of any length: the fixtures repeat under new names", async () => {
    api.reset({ photoCount: 30 });
    const state = await api.openPaths(ARW);
    expect(state.photos).toHaveLength(30);
    expect(new Set(state.photos.map((p) => p.file_name)).size).toBe(30);
    expect(state.photos[11]!.file_name).toBe("DSC04212.ARW");
    expect(state.photos[12]!.file_name).toBe("DSC04213.ARW");
    expect(state.photos[29]!.file_name).toBe("DSC04230.ARW");
    expect(state.photos.map((p) => p.id)).toEqual(Array.from({ length: 30 }, (_, i) => i + 1));
  });

  it("opens the twelve Lisbon photos from the mockups", async () => {
    const state = await api.openPaths(ARW);
    expect(state.folder_name).toBe("Lisbon");
    expect(state.photos).toHaveLength(12);
    expect(state.photos.map((p) => p.file_name)).toEqual(
      Array.from({ length: 12 }, (_, i) => `DSC042${String(i + 1).padStart(2, "0")}.ARW`),
    );
    const first = state.photos[0]!;
    expect(first.camera).toMatchObject({ body: "Sony A1", profile: "A7R IV", match_kind: { kind: "default" } });
    expect(first).toMatchObject({ iso: 400, shutter_s: 1 / 125, aperture_f: 2.8, width_px: 8640, picked: false, reduced: false });
    expect(first.captured_at.slice(0, 10)).toBe("2026-10-04");
    expect(new Set(state.photos.map((p) => p.id)).size).toBe(12);
    expect(state.current_index).toBe(0);
    expect(state.default_export_folder).toBe("~/Pictures/Lisbon export");
    // Everything is waiting when the call returns; development runs in the background.
    expect(state.photos.every((p) => p.status.kind === "waiting")).toBe(true);
  });

  it("opens a folder, and ignores files that aren't ARW", async () => {
    expect((await api.openPaths(["~/Pictures/notes.txt", "~/Pictures/IMG_1.JPG"])).photos).toHaveLength(0);
    expect((await api.openPaths(["~/Pictures/Lisbon"])).photos).toHaveLength(12);
  });

  it("opens through the dialog, and reports a cancelled dialog as null", async () => {
    api.cancelNextDialog();
    expect(await api.openDialog("files")).toBeNull();
    expect((await api.openDialog("folder"))?.photos).toHaveLength(12);
  });

  it("adding photos that are open already changes nothing, and says how many were", async () => {
    const first = await api.openPaths(ARW);
    expect(first.opened).toEqual({ added: 12, already_open: 0 });
    const again = await api.openPaths(ARW);
    expect(again.photos.map((p) => p.id)).toEqual(first.photos.map((p) => p.id));
    expect(again.opened).toEqual({ added: 0, already_open: 1 });
    expect((await api.openPaths(["~/Pictures/Lisbon"])).opened).toEqual({ added: 0, already_open: 12 });
    // Only an open says what it did.
    expect((await api.sessionState()).opened).toBeUndefined();
  });

  it("adds new files after the others, in name order, and develops only those", async () => {
    await api.openPaths(ARW);
    api.flush();
    const state = await api.openPaths(["~/Pictures/Lisbon/DSC09002.ARW", "~/Pictures/Lisbon/DSC09001.ARW", "~/Pictures/Lisbon/dsc04201.arw", "~/Pictures/x.txt"]);
    expect(state.opened).toEqual({ added: 2, already_open: 1 });
    expect(state.photos.map((p) => p.file_name).slice(11)).toEqual(["DSC04212.ARW", "DSC09001.ARW", "DSC09002.ARW"]);
    expect(state.photos.map((p) => p.id)).toEqual(Array.from({ length: 14 }, (_, i) => i + 1));
    expect(state.current_index).toBe(0);
    // The old photos are ready and stay so; the new ones wait, then develop.
    expect(state.photos.slice(0, 12).every((p) => p.status.kind === "ready")).toBe(true);
    expect(state.photos.slice(12).every((p) => p.status.kind === "waiting")).toBe(true);
    api.flush();
    expect((await api.sessionState()).photos.every((p) => p.status.kind === "ready")).toBe(true);
    // A file name is a photo once.
    expect((await api.openPaths(["~/Pictures/Lisbon/DSC09001.ARW"])).opened).toEqual({ added: 0, already_open: 1 });
  });

  it("opens the paths a test queued for the next dialog, of either kind", async () => {
    await api.openPaths(ARW);
    api.queueChoice("open", ["~/Pictures/New/DSC09500.ARW"]);
    const state = await api.openDialog("files");
    expect(state?.opened).toEqual({ added: 1, already_open: 0 });
    expect(state?.photos).toHaveLength(13);
    // The queue is used up: the next dialog is the fixture folder again.
    expect((await api.openDialog("folder"))?.opened).toEqual({ added: 0, already_open: 12 });
  });

  it("honours picked and current from the options", async () => {
    api.reset({ picked: [2, 3, 4], current: 3 });
    const state = await api.openPaths(ARW);
    expect(state.current_index).toBe(3);
    expect(state.picked).toEqual([3, 4, 5]);
    expect(state.photos.filter((p) => p.picked).map((p) => p.file_name)).toEqual([
      "DSC04203.ARW",
      "DSC04204.ARW",
      "DSC04205.ARW",
    ]);
  });

  it("moves the current photo, picks and unpicks, and rejects unknown ids", async () => {
    await api.openPaths(ARW);
    await api.setCurrent(5);
    await api.setPicked(2, true);
    await api.setPicked(4, true);
    await api.setPicked(2, false);
    const state = await api.sessionState();
    expect(state.current_index).toBe(4);
    expect(state.picked).toEqual([4]);
    expect(state.photos[3]!.picked).toBe(true);
    const error = await rejection(api.setCurrent(999));
    expect(error.scope).toBe("app");
    expect(error.message).toMatch(/no longer open/);
  });

  it("returns copies, so callers can't change the mock's state", async () => {
    await api.openPaths(ARW);
    const state = await api.sessionState();
    state.photos[0]!.picked = true;
    state.photos.length = 0;
    expect((await api.sessionState()).photos).toHaveLength(12);
    expect((await api.sessionState()).photos[0]!.picked).toBe(false);
  });
});

describe("development", () => {
  function listen() {
    const progress: PhotoProgress[] = [];
    const thumbnails: ThumbnailReady[] = [];
    api.on("photo_progress", (e) => progress.push(e));
    api.on("thumbnail_ready", (e) => thumbnails.push(e));
    return { progress, thumbnails };
  }

  const statuses = async () => (await api.sessionState()).photos.map((p) => p.status.kind);

  it("takes about four seconds for twelve photos, one at a time", async () => {
    const { thumbnails } = listen();
    await api.openPaths(ARW);

    await vi.advanceTimersByTimeAsync(1);
    expect((await statuses()).slice(0, 2)).toEqual(["developing", "waiting"]);

    await vi.advanceTimersByTimeAsync(1500);
    const midway = await statuses();
    expect(midway.filter((s) => s === "ready").length).toBeGreaterThanOrEqual(3);
    expect(midway.filter((s) => s === "developing")).toHaveLength(1);
    expect(midway.at(-1)).toBe("waiting");

    await vi.advanceTimersByTimeAsync(2800);
    expect((await statuses()).every((s) => s === "ready")).toBe(true);
    expect(thumbnails).toHaveLength(12);
  });

  it("emits photo_progress through decode, estimate and thumbnail for each photo", async () => {
    const { progress } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    const forFirst = progress.filter((e) => e.id === 1).map((e) => `${e.stage}:${e.done}`);
    expect(forFirst).toEqual(["decode:false", "decode:true", "estimate:false", "estimate:true", "thumbnail:true"]);
    expect(progress.every((e) => e.error === null)).toBe(true);
  });

  it("announces each thumbnail by its address: a PNG of at most 240 px is behind it", async () => {
    const { thumbnails } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    expect(thumbnails.map((t) => t.id)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
    for (const { url } of thumbnails) {
      // The picture does not travel in the event (the shell serves it on `ym-frame://`); the mock's address holds it.
      expect(url).toMatch(/^(blob:|data:image\/png;base64,)/);
      expect(url.length).toBeLessThan(2 ** 20);
      if (url.startsWith("data:")) {
        const bytes = Uint8Array.from(atob(url.split(",")[1] ?? ""), (c) => c.charCodeAt(0));
        expect(Array.from(bytes.slice(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
        expect(new DataView(bytes.buffer).getUint32(16)).toBeLessThanOrEqual(240);
      }
    }
  });

  it("gives the thumbnails again on request, as a webview that reloaded needs them", async () => {
    const { thumbnails } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    const again = await api.thumbnails();
    expect(again.map((t) => t.id)).toEqual(thumbnails.map((t) => t.id));
    expect(again.map((t) => t.url)).toEqual(thumbnails.map((t) => t.url));
    await api.closeSession();
    expect(await api.thumbnails()).toEqual([]);
  });

  it("refuses to restart while an export is running, as the shell does", async () => {
    api.reset({ exportHoldAt: 1 });
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    await api.exportStart(job("all"));
    expect(await api.exportBusy()).toBe(true);
    await expect(api.restartApp()).rejects.toMatchObject({ scope: "export" });
    expect(api.system.restarts).toBe(0);
    api.flush();
    await vi.advanceTimersByTimeAsync(1);
    expect(await api.exportBusy()).toBe(false);
    await api.restartApp();
    expect(api.system.restarts).toBe(1);
  });

  it("marks files that can't be developed as skipped, with the reason", async () => {
    const reason = "ARW 6.0 compressed files from the A7 V and A7R VI aren't supported yet.";
    api.reset({ skipped: { "DSC04207.ARW": reason } });
    const { progress, thumbnails } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    const photos = (await api.sessionState()).photos;
    expect(photos[6]!.status).toEqual({ kind: "skipped", reason });
    expect(photos.filter((p) => p.status.kind === "ready")).toHaveLength(11);
    expect(thumbnails.some((t) => t.id === 7)).toBe(false);
    expect(progress.find((e) => e.id === 7 && e.error)?.error).toBe(reason);
  });

  it("speed scales the timers, and Infinity finishes at once", async () => {
    api.reset({ speed: 10 });
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(450);
    expect((await statuses()).every((s) => s === "ready")).toBe(true);

    api.reset({ speed: Infinity });
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(1);
    // Every timer is 0 ms, so a few macrotask turns finish the work.
    await vi.advanceTimersByTimeAsync(50);
    expect((await statuses()).every((s) => s === "ready")).toBe(true);
  });

  it("flush() finishes development immediately", async () => {
    const { thumbnails } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(1);
    api.flush();
    expect((await statuses()).every((s) => s === "ready")).toBe(true);
    expect(thumbnails).toHaveLength(12);
  });

  it("stops listening when unsubscribed", async () => {
    const seen: PhotoProgress[] = [];
    const stop = api.on("photo_progress", (e) => seen.push(e));
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(1);
    const count = seen.length;
    expect(count).toBeGreaterThan(0);
    stop();
    stop(); // safe to call twice
    await vi.advanceTimersByTimeAsync(5000);
    expect(seen).toHaveLength(count);
  });
});

describe("settings", () => {
  beforeEach(async () => {
    await api.openPaths(ARW);
  });

  it("starts at Auto everywhere, resolved against the estimates", async () => {
    const r = await api.getSettings(1);
    expect(r.photo_id).toBe(1);
    expect(r.auto).toEqual({
      exposure: true,
      contrast: true,
      highlights: true,
      shadows: true,
      black: true,
      balance_shadows: true,
      balance_highlights: true,
      saturation: true,
      light_source: true,
      white_balance: true,
      sharpen: true,
      noise_reduction: true,
    });
    expect(r.values).toMatchObject({ exposure_ev: 0.3, kelvin: 5450, tint: 0, contrast: 0, saturation: 0 });
    expect(r.values.sharpen).toEqual({ amount: 120, radius_px: 0.8 });
    expect(r.values.noise_reduction).toEqual({ luminance: 30, color: 50, detail: 50 });
    expect(r.estimates).toEqual(MOCK_ESTIMATES);
    expect(r.settings.exposure).toEqual({ kind: "auto" });
    expect(r.settings).toEqual(defaultSettings());
  });

  it("turns a field Manual when patched, and keeps the other fields Auto", async () => {
    const r = await api.updateSettings(1, { exposure: { kind: "manual", ev: -0.5 }, contrast: manual(12) });
    expect(r.settings.exposure).toEqual({ kind: "manual", ev: -0.5 });
    expect(r.values.exposure_ev).toBe(-0.5);
    expect(r.values.contrast).toBe(12);
    expect(r.auto.exposure).toBe(false);
    expect(r.auto.white_balance).toBe(true);
    // Persisted: a fresh read agrees, and another photo is untouched.
    expect((await api.getSettings(1)).values.exposure_ev).toBe(-0.5);
    expect((await api.getSettings(2)).values.exposure_ev).toBe(0.3);
  });

  it("clamps values to the ranges of docs/01", async () => {
    const r = await api.updateSettings(1, {
      exposure: { kind: "manual", ev: 9 },
      white_balance: { kind: "manual", kelvin: 500, tint: 80 },
      sharpen: { kind: "manual", amount: 999, radius_px: 0 },
      contrast: manual(-400),
      look_intensity: 250,
      rolloff: { highlights: 3, low_light: -1 },
    });
    expect(r.values.exposure_ev).toBe(4);
    expect([r.values.kelvin, r.values.tint]).toEqual([2000, 50]);
    expect(r.values.sharpen).toEqual({ amount: 300, radius_px: 0.3 });
    expect(r.values.contrast).toBe(-100);
    expect(r.values.look_intensity).toBe(100);
    expect(r.values.rolloff).toEqual({ highlights: 1, low_light: 0 });
  });

  it("keeps the output sharpening as a plain number: 100 by default, clamped to 0 to 200, kept by Re-run", async () => {
    const before = await api.getSettings(1);
    expect(before.settings.output_sharpening).toBe(100);
    expect(before.values.output_sharpening).toBe(100);
    expect(defaultSettings().output_sharpening).toBe(100);
    const set = await api.updateSettings(1, { output_sharpening: 35 });
    expect(set.settings.output_sharpening).toBe(35);
    expect(set.values.output_sharpening).toBe(35);
    // Another photo is untouched.
    expect((await api.getSettings(2)).values.output_sharpening).toBe(100);
    expect((await api.updateSettings(1, { output_sharpening: 900 })).values.output_sharpening).toBe(200);
    expect((await api.updateSettings(1, { output_sharpening: -4 })).values.output_sharpening).toBe(0);
    // Re-run resets exposure, white balance and noise: not this.
    await api.updateSettings(1, { output_sharpening: 60 });
    expect((await api.rerunAuto(1)).values.output_sharpening).toBe(60);
  });

  it("rerunAuto resets exposure, white balance and noise to Auto, and leaves contrast and saturation without Apple Auto", async () => {
    await api.updateSettings(1, {
      exposure: { kind: "manual", ev: 1 },
      light_source: "tungsten",
      white_balance: { kind: "manual", kelvin: 3200, tint: 5 },
      noise_reduction: { kind: "manual", luminance: 80, color: 80, detail: 10 },
      contrast: manual(20),
      saturation: manual(-10),
      brightness: 20,
    });
    const r = await api.rerunAuto(1);
    expect(r.auto).toEqual({
      exposure: true,
      contrast: false,
      highlights: true,
      shadows: true,
      black: true,
      balance_shadows: true,
      balance_highlights: true,
      saturation: false,
      light_source: true,
      white_balance: true,
      sharpen: true,
      noise_reduction: true,
    });
    expect([r.values.contrast, r.values.saturation]).toEqual([20, -10]);
    expect(r.values.brightness).toBe(20); // other controls are not touched
  });

  it("emits render_stats after a change", async () => {
    const stats: unknown[] = [];
    api.on("render_stats", (e) => stats.push(e));
    await api.updateSettings(1, { contrast: manual(5) });
    await vi.advanceTimersByTimeAsync(1);
    expect(stats).toEqual([{ ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null }]);
  });

  it("rejects an unknown photo", async () => {
    expect((await rejection(api.getSettings(77))).scope).toBe("app");
  });
});

describe("looks", () => {
  it("lists Neutral (default), Signature and the imported looks of photos.html", async () => {
    const looks = await api.looksList();
    expect(looks.map((l) => l.name)).toEqual([
      "Neutral",
      "Signature",
      "Film warm",
      "Film cool",
      "Bleach bypass",
      "Teal and orange",
      "Soft print",
      "Night blue",
      "Monochrome",
    ]);
    expect(looks.filter((l) => l.is_default).map((l) => l.id)).toEqual(["builtin:neutral"]);
    expect(looks.filter((l) => l.group.kind === "built_in")).toHaveLength(2);
    expect(looks.every((l) => !l.missing)).toBe(true);
    expect(looks[2]).toMatchObject({ id: "arri:3000:film-warm", group: { kind: "arri", no: 3000 }, input: "logc4" });
  });

  it("says which library, kind of file and table size each look has", async () => {
    const looks = await api.looksList();
    expect(looks.slice(0, 2).map((l) => [l.library, l.format, l.lut_size])).toEqual([
      ["built_in", null, null],
      ["built_in", null, null],
    ]);
    // ARRI's LogC4 library, as 65-point 3D LUTs.
    expect(looks.slice(2).every((l) => l.library === "arri_logc4" && l.format === "cube" && l.lut_size === 65)).toBe(true);
  });

  it("has looks in every section with mockLibraries=all, with ARRI's titles repeating from one library to the next", async () => {
    api.reset({ allLibraries: true });
    const looks = await api.looksList();
    const count = (library: string): number => looks.filter((l) => l.library === library).length;
    expect(["built_in", "arri_logc4", "arri_logc3", "arri_logc3_to_rec709", "arri_alf2", "user"].map(count)).toEqual([2, 7, 5, 4, 6, 5]);
    expect(new Set(looks.map((l) => l.id)).size).toBe(looks.length);
    // Every look is where the core's rule puts it.
    for (const look of looks) expect(libraryOf(look.group, look.input, look.format)).toBe(look.library);
    // The same title in LogC4, LogC3 and LogC3 to Rec.709, and in the person's own LUTs.
    const titled = (name: string): string[] => looks.filter((l) => l.name === name).map((l) => l.library);
    expect(titled("Film warm")).toEqual(["arri_logc4", "arri_logc3", "arri_logc3_to_rec709"]);
    expect(titled("Film cool")).toEqual(["arri_logc4", "arri_logc3", "arri_logc3_to_rec709", "user"]);
    // The ALF2 library has three strengths of a look, as 33-point look files.
    expect(looks.filter((l) => l.library === "arri_alf2").map((l) => [l.name, l.format, l.lut_size])).toEqual(
      ["Film warm", "Soft print"].flatMap((t) => ["Low", "Medium", "High"].map((s) => [`${t} · ${s}`, "aml", 33])),
    );
    // The person's own LUTs hold one of each kind of file.
    expect(looks.filter((l) => l.library === "user").map((l) => l.format)).toEqual(["cube", "cube", "alf4", "alf4c", "aml"]);
    expect(looks.filter((l) => l.library === "user").every((l) => l.group.kind === "user")).toBe(true);
  });

  it("puts the looks of a LogC3 package in their own library, under the same titles", async () => {
    api.reset({ looksImported: false });
    for (const zip of ["ARRI_LookLibrary_LogC3.zip", "ARRI_LookLibrary_LogC3_to_Rec709.zip"]) {
      const importing = api.looksImport(`~/Downloads/${zip}`);
      await vi.advanceTimersByTimeAsync(4000);
      await importing;
    }
    const looks = await api.looksList();
    const only = (library: string) => looks.filter((l) => l.library === library);
    expect(only("arri_logc3")).toHaveLength(87);
    expect(only("arri_logc3_to_rec709")).toHaveLength(87);
    expect(only("arri_logc3")[0]).toMatchObject({ id: "arri:3000:film-warm-logc3", name: "Film warm", input: "logc3_ei800", format: "cube", lut_size: 33 });
    expect(only("arri_logc3_to_rec709")[0]).toMatchObject({ id: "arri:3000:film-warm-logc3-rec709", name: "Film warm", input: "logc3_to_rec709" });
  });

  it("files the person's own LUTs and look files under Your LUTs, with their kind of file", async () => {
    api.reset({ looksImported: false });
    const own = async (name: string, input?: "logc4"): Promise<void> => {
      const importing = api.looksImport(`~/Downloads/${name}`, input);
      await vi.advanceTimersByTimeAsync(500);
      await importing;
    };
    await own("Warm_Film.cube", "logc4");
    await own("Warm_Pull.alf4");
    await own("Custom_Look.alf4c");
    const mine = (await api.looksList()).filter((l) => l.library === "user");
    expect(mine.map((l) => [l.name, l.format, l.lut_size])).toEqual([
      ["Warm Film", "cube", 33],
      ["Warm Pull", "alf4", 33],
      ["Custom Look", "alf4c", 33],
    ]);
  });

  it("classifies a look the way the core does", () => {
    const arri = { kind: "arri", no: 3000, name: "Film" } as const;
    expect(libraryOf({ kind: "built_in" }, "logc3_to_rec709", "aml")).toBe("built_in");
    expect(libraryOf({ kind: "user" }, "logc3_to_rec709", "aml")).toBe("user");
    expect(libraryOf(arri, "logc4", "cube")).toBe("arri_logc4");
    expect(libraryOf(arri, "logc4", "alf4")).toBe("arri_logc4");
    expect(libraryOf(arri, "logc4", "alf4c")).toBe("arri_logc4");
    expect(libraryOf(arri, "logc3_ei800", "cube")).toBe("arri_logc3");
    expect(libraryOf(arri, "logc3_to_rec709", "cube")).toBe("arri_logc3_to_rec709");
    expect(libraryOf(arri, "logc3_to_rec709", "aml")).toBe("arri_alf2");
    expect(libraryOf(arri, "logc3_to_rec709", null)).toBe("arri_logc3_to_rec709");
  });

  it("imports the looks when they aren't there yet, and only once", async () => {
    api.reset({ looksImported: false });
    expect(await api.looksList()).toHaveLength(2);
    const events: unknown[] = [];
    api.on("looks_changed", (e) => events.push(e));
    const zip = "~/Downloads/ARRI_LookLibrary_LogC4.zip";
    const importing = api.looksImport(zip);
    await vi.advanceTimersByTimeAsync(3500);
    const first = await importing;
    expect(first.added).toHaveLength(87);
    expect(first.unchanged).toBe(0);
    expect(await api.looksList()).toHaveLength(89);
    await vi.advanceTimersByTimeAsync(1);
    expect(events).toEqual([null]);
    const again = api.looksImport(zip);
    await vi.advanceTimersByTimeAsync(3500);
    expect(await again).toMatchObject({ added: [], unchanged: 87 });
    expect(await api.looksList()).toHaveLength(89);
  });

  it("sets the default look, which new sessions then use", async () => {
    await api.looksSetDefault("arri:3000:film-cool");
    expect((await api.looksList()).filter((l) => l.is_default).map((l) => l.name)).toEqual(["Film cool"]);
    expect((await api.preferencesGet()).default_look).toBe("arri:3000:film-cool");
    await api.openPaths(ARW);
    expect((await api.getSettings(1)).values.look).toBe("arri:3000:film-cool");
    expect((await rejection(api.looksSetDefault("arri:nope"))).scope).toBe("looks");
  });

  it("removes imported looks but not built-in ones, and falls back to Neutral", async () => {
    await api.openPaths(ARW);
    await api.applyLook("all", "arri:3000:night-blue", 100);
    await api.looksSetDefault("arri:3000:night-blue");
    await api.looksRemove("arri:3000:night-blue");
    const looks = await api.looksList();
    expect(looks.some((l) => l.id === "arri:3000:night-blue")).toBe(false);
    expect(looks.find((l) => l.is_default)?.id).toBe("builtin:neutral");
    expect((await api.getSettings(3)).values.look).toBe("builtin:neutral");
    expect((await rejection(api.looksRemove("builtin:signature"))).message).toMatch(/can't be removed/);
  });

  describe("apply_look", () => {
    beforeEach(async () => {
      api.reset({ picked: [2, 3], current: 3 });
      await api.openPaths(ARW);
    });
    const lookOf = async (id: number) => (await api.getSettings(id)).values;

    it("applies to this photo only", async () => {
      await api.applyLook("this", "builtin:signature", 60);
      expect(await lookOf(4)).toMatchObject({ look: "builtin:signature", look_intensity: 60 });
      expect(await lookOf(3)).toMatchObject({ look: "builtin:neutral", look_intensity: 100 });
    });

    it("applies to the picked photos", async () => {
      await api.applyLook("picked", "arri:3000:film-warm", 80);
      expect((await lookOf(3)).look).toBe("arri:3000:film-warm");
      expect((await lookOf(4)).look).toBe("arri:3000:film-warm");
      expect((await lookOf(1)).look).toBe("builtin:neutral");
    });

    it("applies to all photos, clamping the intensity", async () => {
      await api.applyLook("all", "builtin:signature", 140);
      for (let id = 1; id <= 12; id++) expect(await lookOf(id)).toMatchObject({ look: "builtin:signature", look_intensity: 100 });
    });

    it("rejects a look that isn't in the library", async () => {
      const error = await rejection(api.applyLook("all", "arri:3000:gone", 100));
      expect(error.scope).toBe("looks");
      expect(error.message).toMatch(/Import it again/);
    });
  });
});

describe("export", () => {
  beforeEach(async () => {
    api.reset({ picked: [2, 3, 4], current: 3 });
    await api.openPaths(ARW);
  });

  it("estimates size and time with the arithmetic of the export mockup", async () => {
    // JPEG at 92: 6 + 42 × 0.45 = 24.9 MB a photo, three picked photos.
    const jpeg = await api.exportEstimate(job("picked"));
    expect(jpeg.bytes).toBe(Math.round(24.9 * 3 * 1e6));
    expect(jpeg.seconds).toBeCloseTo(3.8, 1);
    // 16-bit LZW TIFF: 298 × 0.68 MB; DNG: 300 MB; PNG 16-bit: 210 MB.
    expect((await api.exportEstimate(job("this", { format: "tiff" }))).bytes).toBe(Math.round(298 * 0.68 * 1e6));
    expect((await api.exportEstimate(job("this", { format: "dng" }))).bytes).toBe(300e6);
    expect((await api.exportEstimate(job("this", { format: "png" }))).bytes).toBe(210e6);
    // Size by long edge scales with the area; DNG ignores it.
    const half = await api.exportEstimate(job("this", { format: "png", size: { kind: "long_edge", px: 4320 } }));
    expect(half.bytes).toBe(Math.round(210e6 / 4));
    const dng = await api.exportEstimate(job("this", { format: "dng", size: { kind: "long_edge", px: 4320 } }));
    expect(dng.bytes).toBe(300e6);
    // 12 photos in all.
    expect((await api.exportEstimate(job("all"))).bytes).toBe(Math.round(24.9 * 12 * 1e6));
  });

  it("runs the picked photos in about four seconds, with per-file progress", async () => {
    const events: ExportProgress[] = [];
    api.on("export_progress", (e) => events.push(e));
    const id = await api.exportStart(job("picked"));
    expect(id).toBe(1);

    await vi.advanceTimersByTimeAsync(1);
    expect(events[0]).toMatchObject({ job_id: 1, file: "DSC04203_neutral.jpg", done: 0, total: 3, file_percent: 0, error: null });

    await vi.advanceTimersByTimeAsync(1800); // 18 ticks: 1.44 photos
    const mid = events.at(-1)!;
    expect(mid).toMatchObject({ file: "DSC04204_neutral.jpg", done: 1, total: 3 });
    expect(mid.file_percent).toBeGreaterThan(0);
    expect(mid.file_percent).toBeLessThan(100);

    await vi.advanceTimersByTimeAsync(2500);
    expect(events.at(-1)).toMatchObject({ file: "DSC04205_neutral.jpg", done: 3, total: 3, file_percent: 100 });
    const count = events.length;
    await vi.advanceTimersByTimeAsync(5000);
    expect(events).toHaveLength(count); // finished work stops emitting
  });

  it("names files from the template and the format", async () => {
    const files: string[] = [];
    api.on("export_progress", (e) => files.push(e.file));
    await api.applyLook("this", "arri:3000:film-warm", 100);
    await api.exportStart(job("this", { format: "tiff", name_template: "{date}_{name}_{look}_{seq}" }));
    await vi.advanceTimersByTimeAsync(1);
    expect(files[0]).toBe("2026-10-04_DSC04204_film-warm_001.tif");
  });

  it("makes the file stem safe, as the core does", async () => {
    const files: string[] = [];
    api.on("export_progress", (e) => files.push(e.file));
    await api.exportStart(job("this", { name_template: "{name}*final?" }));
    await vi.advanceTimersByTimeAsync(1);
    expect(files[0]).toBe("DSC04204_final_.jpg");
  });

  it("cancels a running job", async () => {
    const events: ExportProgress[] = [];
    api.on("export_progress", (e) => events.push(e));
    const id = await api.exportStart(job("all"));
    await vi.advanceTimersByTimeAsync(500);
    await api.exportCancel(id);
    const count = events.length;
    await vi.advanceTimersByTimeAsync(20_000);
    expect(events).toHaveLength(count);
    await api.exportCancel(id); // cancelling again is harmless
  });

  it("numbers jobs, and flush() finishes them", async () => {
    const events: ExportProgress[] = [];
    api.on("export_progress", (e) => events.push(e));
    expect(await api.exportStart(job("this"))).toBe(1);
    expect(await api.exportStart(job("this"))).toBe(2);
    await vi.advanceTimersByTimeAsync(1);
    api.flush();
    expect(events.filter((e) => e.done === e.total).map((e) => e.job_id).sort()).toEqual([1, 2]);
  });

  it("refuses to start when nothing is picked", async () => {
    api.reset({});
    await api.openPaths(ARW);
    const error = await rejection(api.exportStart(job("picked")));
    expect(error).toEqual({
      scope: "export",
      message: "No photos are picked. Pick some photos, or choose All, then export again.",
    });
  });

  it("says whether an export is busy: by command, and by export_busy when the queue goes busy and quiet", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const busy: boolean[] = [];
    api.on("export_busy", (e) => busy.push(e.busy));
    await api.openPaths(ARW);
    api.flush();
    await expect(api.exportBusy()).resolves.toBe(false);
    const id = await api.exportStart(job("all"));
    await Promise.resolve();
    await expect(api.exportBusy()).resolves.toBe(true);
    expect(busy).toEqual([true]);
    api.flush();
    await Promise.resolve();
    await expect(api.exportBusy()).resolves.toBe(false);
    expect(busy).toEqual([true, false]);
    // A cancelled run is quiet too.
    const second = await api.exportStart(job("this"));
    await api.exportCancel(second);
    await Promise.resolve();
    expect(busy).toEqual([true, false, true, false]);
    expect(id).not.toBe(second);
  });

  it("writes the equivalent command line, as the mockup does", async () => {
    expect(await api.cliPreview(job("picked"))).toBe(
      [
        "ym render DSC04203.ARW DSC04204.ARW DSC04205.ARW \\",
        "  --look neutral \\",
        "  --format jpeg \\",
        "  --quality 92 \\",
        "  --colour-space srgb \\",
        '  --out "~/Pictures/Lisbon export"',
      ].join("\n"),
    );
    const tiff = await api.cliPreview(
      job("all", {
        format: "tiff",
        bit_depth: 8,
        tiff_compression: "zip",
        color_space: "display_p3",
        size: { kind: "long_edge", px: 3000 },
        destination: "~/Exports",
        if_exists: "skip",
      }),
    );
    expect(tiff).toContain("ym render ~/Pictures/Lisbon \\");
    for (const flag of ["--format tiff", "--bits 8", "--compression zip", "--colour-space p3", "--long-edge 3000", "--skip-existing", '--out "~/Exports"']) {
      expect(tiff).toContain(flag);
    }
    const dng = await api.cliPreview(job("this", { format: "dng" }));
    expect(dng).not.toContain("--colour-space");
    expect(dng).not.toContain("--quality");
  });
});

describe("preferences", () => {
  it("has the defaults of docs/09 and docs/12", async () => {
    const p = await api.preferencesGet();
    expect(p.default_look).toBe("builtin:neutral");
    expect(p.display_rendering).toEqual({ kind: "aces2" });
    expect(p.check_for_updates_automatically).toBe(true);
    expect(p.remove_location_from_exports).toBe(false);
    expect(p.export_defaults).toEqual(job("this").options);
  });

  it("applies a patch and returns the new preferences", async () => {
    const next = await api.preferencesSet({ check_for_updates_automatically: false, watch_downloads_during_import: true });
    expect(next.check_for_updates_automatically).toBe(false);
    expect(next.watch_downloads_during_import).toBe(true);
    expect(next.default_look).toBe("builtin:neutral");
    expect(await api.preferencesGet()).toEqual(next);
  });

  it("refuses a default look that doesn't exist", async () => {
    expect((await rejection(api.preferencesSet({ default_look: "user:nope" }))).scope).toBe("preferences");
  });

  it("installs the command-line tool link", async () => {
    expect(await api.installCliLink()).toEqual({ path: "/usr/local/bin/ym" });
  });
});

describe("viewport", () => {
  beforeEach(async () => {
    await api.openPaths(ARW);
  });

  it("records the rect, mode, before state and inputs", async () => {
    const rect = { x: 0, y: 75, width: 1020, height: 680, scale: 2 };
    await api.viewportSetRect(rect);
    await api.viewportSetMode({ kind: "percent", value: 200 });
    await api.setBefore(true);
    await api.viewportInput({ kind: "pan", dx: 4, dy: -2 });
    expect(api.viewport).toEqual({
      rect,
      mode: { kind: "percent", value: 200 },
      before: true,
      inputs: [{ kind: "pan", dx: 4, dy: -2 }],
    });
  });

  it("toggles between fit and 100 % on double click", async () => {
    await api.viewportInput({ kind: "toggle_zoom", x: 10, y: 10 });
    expect(api.viewport.mode).toEqual({ kind: "percent", value: 100 });
    await api.viewportInput({ kind: "toggle_zoom", x: 10, y: 10 });
    expect(api.viewport.mode).toEqual({ kind: "fit" });
  });

  it("samples display values from the scene under the pointer", async () => {
    await api.setCurrent(4); // DSC04204: sky #7FA6C7, near #3F5468
    await api.viewportSetRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
    expect(await api.samplePixel(20, 20)).toEqual({ r: 0x7f, g: 0xa6, b: 0xc7 });
    expect(await api.samplePixel(750, 950)).toEqual({ r: 0x3f, g: 0x54, b: 0x68 });
  });

  it("stores a white pick as a point in image coordinates: the click is found on the picture, not on the paper around it", async () => {
    // A 3:2 photo in a 2:1 cell: the picture spans the middle three quarters across.
    await api.viewportSetRect({ x: 0, y: 0, width: 1000, height: 500, scale: 2 });
    await api.viewportInput({ kind: "pick_white", x: 250, y: 100 });
    const pick = (await api.getSettings(1)).values.white_pick;
    expect(pick?.[0]).toBeCloseTo((0.25 - 0.125) / 0.75, 9);
    expect(pick?.[1]).toBeCloseTo(0.2, 9);
  });

  it("a pick sets the white balance by hand, the light preset to Auto, and says settings_changed", async () => {
    const changed: unknown[] = [];
    api.on("settings_changed", (e) => changed.push(e));
    await api.updateSettings(1, { light_source: "tungsten" });
    await api.viewportSetRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
    await api.viewportInput({ kind: "pick_white", x: 750, y: 40 });
    await Promise.resolve();
    const rs = await api.getSettings(1);
    expect(rs.settings.light_source).toBe("auto");
    expect(rs.settings.white_balance.kind).toBe("manual");
    expect(rs.settings.white_pick).toEqual([0.5, 0.04]);
    expect(changed).toEqual([{ photo_id: 1 }]);
  });

  it("a click on the paper picks nothing and sends nothing", async () => {
    const changed: unknown[] = [];
    api.on("settings_changed", (e) => changed.push(e));
    await api.updateSettings(1, { orientation: 90 });
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    // A portrait photo in a 3:2 cell: the picture spans the middle 44 % across.
    await api.viewportInput({ kind: "pick_white", x: 20, y: 300 });
    await Promise.resolve();
    expect((await api.getSettings(1)).settings.white_pick).toBeNull();
    expect(changed).toEqual([]);
    await api.viewportInput({ kind: "pick_white", x: 450, y: 300 });
    expect((await api.getSettings(1)).settings.white_pick?.map((v) => Math.round(v * 1000) / 1000)).toEqual([0.5, 0.5]);
  });

  it.each([
    [0, [0.3, 0.2]],
    [90, [0.2, 0.7]],
    [180, [0.7, 0.8]],
    [270, [0.8, 0.3]],
  ] as const)("maps a click through a quarter turn of %i° back to the photo as the camera had it", async (orientation, want) => {
    // The picture's (0.3, 0.2): the photo turned clockwise takes (x, y) to (1 − y, x), so (0.3, 0.2) of a 90° picture is (0.2, 0.7) of the photo.
    await api.updateSettings(1, { orientation });
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    const aspect = orientation % 180 === 0 ? 1.5 : 1 / 1.5;
    const width = Math.min(1, aspect / 1.5);
    const height = Math.min(1, 1.5 / aspect);
    const click = { x: 900 * ((1 - width) / 2 + 0.3 * width), y: 600 * ((1 - height) / 2 + 0.2 * height) };
    await api.viewportInput({ kind: "pick_white", ...click });
    const pick = (await api.getSettings(1)).settings.white_pick;
    expect(pick?.[0]).toBeCloseTo(want[0], 6);
    expect(pick?.[1]).toBeCloseTo(want[1], 6);
  });

  it("reads the colour of the spot the click is on, also on a turned photo (the pixel readout and the pick agree with the picture)", async () => {
    await api.setCurrent(4); // DSC04204: sky #7FA6C7, near #3F5468 along the bottom edge of the photo
    await api.updateSettings(4, { orientation: 90 });
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 1 });
    // Turned clockwise, the bottom of the photo is the left edge of the picture (which spans the middle 44 % of the cell).
    expect(await api.samplePixel(450 - 0.22 * 900 + 6, 300)).toEqual({ r: 0x3f, g: 0x54, b: 0x68 });
    expect(await api.samplePixel(450 + 0.22 * 900 - 6, 300)).toMatchObject({ r: 0x7f, g: 0xa6, b: 0xc7 });
  });

  it("maps a click through the crop to the whole photo", async () => {
    await api.updateSettings(1, { crop: { rect: { x: 0.25, y: 0.5, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" } });
    // A crop of half the width and half the height keeps the photo's 3:2 shape, so the cropped picture fills the 3:2 cell.
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    await api.viewportInput({ kind: "pick_white", x: 450, y: 300 });
    const pick = (await api.getSettings(1)).settings.white_pick;
    expect(pick?.[0]).toBeCloseTo(0.5, 6);
    expect(pick?.[1]).toBeCloseTo(0.75, 6);
  });

  it("a pixel readout on the paper fails, as the core's does", async () => {
    await api.updateSettings(1, { orientation: 90 });
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    expect(await rejection(api.samplePixel(20, 300))).toEqual({ scope: "viewport", message: "That point is outside the photo." });
    await expect(api.samplePixel(450, 300)).resolves.toMatchObject({ r: expect.any(Number) });
  });

  it("has no viewport backend unless asked: the page draws the picture itself", async () => {
    await expect(api.viewportInfo()).resolves.toBeNull();
    api.configure({ viewport: "native" });
    await expect(api.viewportInfo()).resolves.toEqual({ backend: "native" });
    api.configure({ viewport: "fallback" });
    await expect(api.viewportInfo()).resolves.toEqual({ backend: "fallback" });
  });

  it("sends no frames as native or without a backend", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, viewport: "native" });
    const frames: unknown[] = [];
    api.on("viewport_frame", (f) => frames.push(f));
    await api.openPaths(ARW);
    api.flush();
    await api.viewportSetRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
    await Promise.resolve();
    expect(frames).toEqual([]);
  });

  describe("as the fallback viewport", () => {
    it("sends a picture of the cell's size when the cell is placed and the photo is ready, and takes it away when the session closes", async () => {
      const api = createMockApi({ speed: Number.POSITIVE_INFINITY, viewport: "fallback" });
      const frames: Array<{ seq: number; width: number; height: number; sharp: boolean; url: string }> = [];
      api.on("viewport_frame", (f) => frames.push(f));
      await api.openPaths(ARW);
      await api.viewportSetRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
      await Promise.resolve();
      // The photo is not ready yet: there is nothing to draw.
      expect(frames).toEqual([]);
      api.flush();
      await Promise.resolve();
      expect(frames).toHaveLength(1);
      expect(frames[0]).toMatchObject({ width: 2000, height: 1332, sharp: true });
      const svg = decodeURIComponent(frames[0]!.url.replace(/^data:image\/svg\+xml;charset=utf-8,/, ""));
      expect(svg).toContain('width="2000" height="1332"');
      expect(svg).toContain('<rect width="100%" height="100%" fill="#FFFFFF"/>');
      // A change of settings, the photo, Before and the zoom each send a new, newer frame.
      await api.updateSettings(1, { contrast: manual(10) });
      await api.setCurrent(2);
      await api.setBefore(true);
      await api.viewportSetMode({ kind: "percent", value: 100 });
      await Promise.resolve();
      expect(frames.map((f) => f.seq)).toEqual([1, 2, 3, 4, 5]);
      await api.closeSession();
      await Promise.resolve();
      expect(frames.at(-1)).toEqual({ seq: 6, width: 0, height: 0, sharp: true, url: "" });
      // Nothing more is said while there is nothing to take away.
      await api.viewportSetRect({ x: 0, y: 75, width: 1010, height: 670, scale: 2 });
      await Promise.resolve();
      expect(frames).toHaveLength(6);
    });

    it("sends no picture for a photo that can't be developed, and takes the previous one away", async () => {
      const api = createMockApi({ speed: Number.POSITIVE_INFINITY, viewport: "fallback", skipped: { DSC04202: "Not readable." } });
      const frames: Array<{ url: string }> = [];
      api.on("viewport_frame", (f) => frames.push(f));
      await api.openPaths(ARW);
      api.flush();
      await api.viewportSetRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
      await Promise.resolve();
      expect(frames.at(-1)?.url).not.toBe("");
      await api.setCurrent(2);
      await Promise.resolve();
      expect(frames.at(-1)?.url).toBe("");
      await api.setCurrent(3);
      await Promise.resolve();
      expect(frames.at(-1)?.url).not.toBe("");
    });
  });

  it("a new photo opens at Fit with the camera view off, as the core's does", async () => {
    await api.viewportSetMode({ kind: "percent", value: 200 });
    await api.setBefore(true);
    await api.setCurrent(2);
    expect(api.viewport.mode).toEqual({ kind: "fit" });
    expect(api.viewport.before).toBe(false);
  });

  it("exposes each photo's SVG fixture", async () => {
    // Small SVGs are inlined as data URIs, which the app's CSP allows (img-src data:).
    const url = api.fixtureUrl(4);
    expect(url).toBeDefined();
    expect(decodeURIComponent(url as string)).toContain("#7FA6C7"); // DSC04204's sky
    expect(api.fixtureUrl(999)).toBeUndefined();
  });
});

describe("test controls", () => {
  it("failNext makes one call reject, then recovers", async () => {
    const error: AppError = { scope: "open", message: "Can't read that folder. Check its permissions and try again." };
    api.failNext("openPaths", error);
    expect(await rejection(api.openPaths(ARW))).toEqual(error);
    expect((await api.openPaths(ARW)).photos).toHaveLength(12);
  });

  it("reset forgets the session and restores looks and preferences", async () => {
    await api.openPaths(ARW);
    await api.preferencesSet({ check_for_updates_automatically: false });
    await api.looksRemove("arri:3000:film-warm");
    api.reset();
    expect((await api.sessionState()).photos).toEqual([]);
    expect((await api.preferencesGet()).check_for_updates_automatically).toBe(true);
    expect(await api.looksList()).toHaveLength(9);
  });

  it("ignores undefined options and a speed that isn't positive", async () => {
    api.reset({ speed: undefined, current: undefined });
    expect(api.options).toMatchObject({ speed: 1, current: 0, looksImported: true });
    api.configure({ speed: 0 });
    expect(api.options.speed).toBe(1);
    api.configure({ speed: -3 });
    expect(api.options.speed).toBe(1);
    api.configure({ speed: 5 });
    expect(api.options.speed).toBe(5);
  });

  it("isolates a listener that throws", async () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const seen: unknown[] = [];
    api.on("render_stats", () => {
      throw new Error("listener bug");
    });
    api.on("render_stats", (e) => seen.push(e));
    await api.openPaths(ARW);
    await api.setCurrent(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(seen).toHaveLength(1);
    expect(error).toHaveBeenCalledTimes(1);
    error.mockRestore();
  });

  it("is deterministic: two instances agree", async () => {
    const other = createMockApi();
    expect(await api.openPaths(ARW)).toEqual(await other.openPaths(ARW));
    expect(await api.getSettings(5)).toEqual(await other.getSettings(5));
    other.reset();
  });

  it("hands listeners copies of event payloads", async () => {
    const seen: ThumbnailReady[] = [];
    api.on("thumbnail_ready", (e) => {
      seen.push(e);
      e.url = "";
    });
    api.on("thumbnail_ready", (e) => seen.push(e));
    await api.openPaths(ARW);
    api.flush();
    expect(seen.filter((e) => e.url.length > 0)).toHaveLength(12);
  });
});

describe("pure helpers", () => {
  it("lookSlug keeps what follows the last colon", () => {
    expect(lookSlug("builtin:neutral")).toBe("neutral");
    expect(lookSlug("arri:3000:film-warm")).toBe("film-warm");
    expect(lookSlug("plain")).toBe("plain");
  });

  it("expandNameTemplate fills known tokens and leaves the rest", () => {
    const parts = { name: "DSC1", look: "neutral", date: "2026-10-04", seq: 7, camera: "sony-a1" };
    expect(expandNameTemplate("{name}_{look}", parts)).toBe("DSC1_neutral");
    expect(expandNameTemplate("{camera}-{date}-{seq}-{other}", parts)).toBe("sony-a1-2026-10-04-007-{other}");
  });

  it("clampSettings leaves Auto and in-range values alone", () => {
    const s = defaultSettings();
    expect(clampSettings(s)).toEqual(s);
  });

  it("resolveSettings fills Auto from the estimates", () => {
    const r = resolveSettings(9, defaultSettings(), MOCK_ESTIMATES);
    expect(r.values.exposure_ev).toBe(MOCK_ESTIMATES.exposure_ev);
    expect(r.photo_id).toBe(9);
  });
});

describe("what the screens need beyond docs/01", () => {
  const open = async (api: ReturnType<typeof createMockApi>) => {
    await api.openDialog("files");
    api.flush();
  };

  it("closes the session: development stops, the photos go, preferences and looks stay", async () => {
    const api = createMockApi({ speed: 1 });
    await api.preferencesSet({ default_look: "builtin:signature" });
    await api.openDialog("files");
    await api.closeSession();
    expect((await api.sessionState()).photos).toEqual([]);
    expect((await api.preferencesGet()).default_look).toBe("builtin:signature");
    expect(await api.looksList()).toHaveLength(9);
    // A new session starts from scratch.
    await api.openDialog("folder");
    expect((await api.sessionState()).photos).toHaveLength(12);
    api.reset();
  });

  it("makes look thumbnails: the address of a PNG of 240 × 160 through the look, different for each look", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const neutral = await api.lookThumbnail(4, "builtin:neutral");
    const teal = await api.lookThumbnail(4, "arri:3000:teal-and-orange");
    // An address, as the shell answers (there `ym-frame://…`; here an object URL, or a data URL where the browser has none).
    expect(neutral).toMatch(/^(blob:|data:image\/png;base64,)/);
    expect(teal).toMatch(/^(blob:|data:image\/png;base64,)/);
    expect(teal).not.toBe(neutral);
    if (neutral.startsWith("data:")) {
      const bytes = Uint8Array.from(atob(neutral.split(",")[1] ?? ""), (c) => c.charCodeAt(0));
      expect(Array.from(bytes.slice(0, 8))).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
      expect(Array.from(bytes.slice(16, 24))).toEqual([0, 0, 0, 240, 0, 0, 0, 160]); // IHDR: width, height
    }
  });

  it("refuses a look thumbnail for a look that is missing or unknown", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, missingLooks: ["film-warm"] });
    await open(api);
    await expect(api.lookThumbnail(4, "arri:3000:film-warm")).rejects.toEqual({ scope: "looks", message: "File missing or changed. Import again." });
    await expect(api.lookThumbnail(4, "arri:3000:nope")).rejects.toMatchObject({ scope: "looks" });
  });

  it("lists a missing look with missing: true", async () => {
    const api = createMockApi({ missingLooks: ["film-warm", "night-blue"] });
    const looks = await api.looksList();
    expect(looks.filter((l) => l.missing).map((l) => l.name)).toEqual(["Film warm", "Night blue"]);
    expect(await api.looksList()).toHaveLength(9);
  });

  it("builds a histogram that follows exposure", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const flat = await api.histogram(4);
    expect(flat.red).toHaveLength(64);
    expect(Math.max(...flat.red, ...flat.green, ...flat.blue)).toBe(1);
    await api.updateSettings(4, { exposure: { kind: "manual", ev: 2 } });
    const bright = await api.histogram(4);
    expect(bright.green).not.toEqual(flat.green);
    expect(await api.histogram(4)).toEqual(bright);
  });

  it("keeps the histogram of the finished picture while Before is on", async () => {
    // The core builds the histogram from the graded picture and never reads the Before state (docs/02).
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const after = await api.histogram(4);
    await api.setBefore(true);
    expect(await api.histogram(4)).toEqual(after);
    await api.setBefore(false);
    expect(await api.histogram(4)).toEqual(after);
  });

  it("resolves a light preset to a kelvin while white balance is Auto, and a manual kelvin wins", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    let rs = await api.updateSettings(4, { light_source: "tungsten" });
    expect([rs.values.kelvin, rs.auto.white_balance, rs.auto.light_source]).toEqual([3200, true, false]);
    rs = await api.updateSettings(4, { white_balance: { kind: "manual", kelvin: 6000, tint: 2 } });
    expect([rs.values.kelvin, rs.values.tint]).toEqual([6000, 2]);
    rs = await api.updateSettings(4, { white_balance: { kind: "auto" }, light_source: "auto" });
    expect(rs.values.kelvin).toBe(5450);
  });

  it("resolves a picked white to a manual white balance from the colour under the point", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const sky = await api.updateSettings(4, { white_pick: [0.5, 0.05] });
    expect(sky.settings.white_pick).toEqual([0.5, 0.05]);
    expect(sky.settings.white_balance.kind).toBe("manual");
    // The sky of DSC04204 is blue: calling it neutral means the light is cooler (a higher kelvin) than Automatic assumed.
    expect(sky.values.kelvin).toBeGreaterThan(5450);
    // The ground of DSC04201 is dark and bluish too; a warm scene (DSC04202's sky) gives a lower kelvin.
    const warm = await api.updateSettings(2, { white_pick: [0.5, 0.05] });
    expect(warm.values.kelvin).toBeLessThan(5450);
    // Clearing the pick leaves the white balance as it was.
    const cleared = await api.updateSettings(4, { white_pick: null });
    expect(cleared.settings.white_pick).toBeNull();
    expect(cleared.settings.white_balance.kind).toBe("manual");
  });

  it("re-sends a photo's thumbnail after its look changed, once for a burst of changes", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const seen: number[] = [];
    api.on("thumbnail_ready", (e) => seen.push(e.id));
    await api.updateSettings(4, { look: "arri:3000:teal-and-orange" });
    await api.updateSettings(4, { look_intensity: 50 });
    await api.updateSettings(4, { contrast: manual(3) });
    await vi.advanceTimersByTimeAsync(200);
    expect(seen).toEqual([4]);
  });

  it("opens photos with the initial settings and summary overrides given", async () => {
    const api = createMockApi({
      speed: Number.POSITIVE_INFINITY,
      initialSettings: { DSC04204: { highlights: manual(-18), saturation: manual(6) } },
      photoOverrides: { DSC04204: { reduced: true, camera: { match_kind: { kind: "sensor_family", via: "Sony A7 IV" } } } },
    });
    await open(api);
    const rs = await api.getSettings(4);
    expect([rs.values.highlights, rs.values.saturation]).toEqual([-18, 6]);
    expect((await api.getSettings(5)).values.highlights).toBe(0);
    const photo = (await api.sessionState()).photos[3];
    expect(photo?.reduced).toBe(true);
    expect(photo?.camera.match_kind).toEqual({ kind: "sensor_family", via: "Sony A7 IV" });
    expect(photo?.camera.body).toBe("Sony A1");
  });

  it("holds development at a fraction of the photos and resumes on flush", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY, holdAt: 5.4 });
    await api.openDialog("files");
    await vi.advanceTimersByTimeAsync(200);
    const kinds = (await api.sessionState()).photos.map((p) => p.status.kind);
    expect(kinds.filter((k) => k === "ready")).toHaveLength(5);
    expect(kinds.filter((k) => k === "developing")).toHaveLength(1);
    expect(kinds.filter((k) => k === "waiting")).toHaveLength(6);
    api.flush();
    expect((await api.sessionState()).photos.every((p) => p.status.kind === "ready")).toBe(true);
  });

  it("reports render_stats when the photo cell is placed", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const stats: unknown[] = [];
    api.on("render_stats", (s) => stats.push(s));
    await api.viewportSetRect({ x: 0, y: 75, width: 1000, height: 666, scale: 2 });
    await Promise.resolve();
    expect(stats).toEqual([{ ms: 42, path: "neural", space: "display_p3", zoom_percent: null, fit_percent: null }]);
  });

  it("clamps a crop, an orientation and a colour balance point", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    await open(api);
    const rs = await api.updateSettings(4, {
      crop: { rect: { x: -1, y: 0.5, width: 5, height: 5 }, angle_deg: 90, aspect: "free" },
      orientation: 450 as never,
      color_balance: { shadows: { kind: "manual", point: [3, 4] }, midtones: [0, 0], highlights: { kind: "auto" } },
    });
    expect(rs.settings.crop?.angle_deg).toBe(45);
    expect(rs.settings.crop?.rect).toEqual({ x: 0, y: 0.5, width: 1, height: 0.5 });
    expect(rs.settings.orientation).toBe(90);
    expect(Math.hypot(...rs.values.color_balance.shadows)).toBeCloseTo(1, 6);
    expect(rs.auto.balance_shadows).toBe(false);
  });
});

describe("optionsFromLocation", () => {
  it("reads every option the end-to-end tests set from the URL", async () => {
    const { optionsFromLocation } = await import("./api.mock");
    expect(
      optionsFromLocation("?mockSpeed=Infinity&mockSkipped=DSC04207:Bad file.,DSC04209&mockPicked=2,3,4&mockCurrent=3&mockLooks=0&mockMissing=film-warm&mockHold=5.4"),
    ).toEqual({
      speed: Number.POSITIVE_INFINITY,
      skipped: { DSC04207: "Bad file.", DSC04209: "This file is from a camera mode we can't read yet." },
      picked: [2, 3, 4],
      current: 3,
      looksImported: false,
      missingLooks: ["film-warm"],
      holdAt: 5.4,
    });
    expect(optionsFromLocation("?mockPhotos=500")).toEqual({ photoCount: 500 });
    expect(optionsFromLocation("?mockLibraries=all")).toEqual({ allLibraries: true });
    expect(optionsFromLocation("?mockLibraries=none")).toEqual({});
    expect(optionsFromLocation("?mockViewport=fallback")).toEqual({ viewport: "fallback" });
    expect(optionsFromLocation("?mockViewport=native")).toEqual({ viewport: "native" });
    expect(optionsFromLocation("?mockViewport=metal")).toEqual({});
    expect(optionsFromLocation("?mockPhotos=0")).toEqual({});
    expect(optionsFromLocation("")).toEqual({});
    expect(optionsFromLocation("?mockSpeed=-3&mockHold=x")).toEqual({});
    expect(optionsFromLocation(undefined as never)).toEqual({});
  });
});

describe("copy and paste settings", () => {
  const ready = async (): Promise<void> => {
    await api.openPaths(ARW);
    api.flush();
  };

  it("copies every setting but the crop, the orientation and the white pick; Auto stays Auto", async () => {
    await ready();
    await api.updateSettings(1, {
      contrast: manual(20),
      exposure: { kind: "manual", ev: 1 },
      crop: { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" },
      orientation: 90,
      white_pick: [0.5, 0.5],
    });
    const patch = await api.copySettings(1);
    expect(patch).toMatchObject({ contrast: manual(20), exposure: { kind: "manual", ev: 1 }, sharpen: { kind: "auto" }, noise_reduction: { kind: "auto" }, light_source: "auto" });
    for (const key of ["crop", "orientation", "white_pick"]) expect(patch).not.toHaveProperty(key);
    expect(Object.keys(patch).length).toBe(Object.keys(defaultSettings()).length - 3);
  });

  it("turns a white balance that came from a picked white into the manual Kelvin and tint it resolved to", () => {
    const rs = resolveSettings(1, { ...defaultSettings(), white_pick: [0.5, 0.5] }, MOCK_ESTIMATES);
    expect(copiedPatch(rs).white_balance).toEqual({ kind: "manual", kelvin: rs.values.kelvin, tint: rs.values.tint });
    // A light preset, a manual white balance or no pick are copied as they are.
    const preset = resolveSettings(1, { ...defaultSettings(), white_pick: [0.5, 0.5], light_source: "tungsten" }, MOCK_ESTIMATES);
    expect(copiedPatch(preset).white_balance).toEqual({ kind: "auto" });
    expect(copiedPatch(preset).light_source).toBe("tungsten");
    const manualWhite = resolveSettings(1, { ...defaultSettings(), white_pick: [0.5, 0.5], white_balance: { kind: "manual", kelvin: 4100, tint: 3 } }, MOCK_ESTIMATES);
    expect(copiedPatch(manualWhite).white_balance).toEqual({ kind: "manual", kelvin: 4100, tint: 3 });
    expect(copiedPatch(resolveSettings(1, defaultSettings(), MOCK_ESTIMATES)).white_balance).toEqual({ kind: "auto" });
  });

  it("refuses to copy a photo that is not developed or cannot be opened", async () => {
    api.reset({ skipped: { DSC04203: "This file is not a raw file." } });
    await api.openPaths(ARW);
    expect((await rejection(api.copySettings(1))).message).toMatch(/still developing/);
    api.flush();
    expect((await rejection(api.copySettings(3))).message).toMatch(/can't be opened/);
    expect(await api.copySettings(2)).toBeDefined();
    expect((await rejection(api.copySettings(99))).scope).toBe("app");
  });

  it("pastes onto each photo, leaving its crop, orientation and white pick, and answers with one step per photo", async () => {
    await ready();
    await api.updateSettings(1, { contrast: manual(20), look: "builtin:signature" });
    await api.updateSettings(2, { contrast: manual(-5), crop: { rect: { x: 0, y: 0, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" }, orientation: 180 });
    const patch = await api.copySettings(1);
    const events: number[] = [];
    api.on("settings_changed", (e) => events.push(e.photo_id));
    const report = await api.pasteSettings([2, 3, 2, 99], { ...patch, crop: null, orientation: 270, white_pick: [0.1, 0.1] });
    expect(report.skipped).toEqual([99]);
    expect(report.pasted.map((p) => p.id)).toEqual([2, 3]);
    expect(report.pasted[0]).toMatchObject({ before: { contrast: manual(-5), look: "builtin:neutral" }, after: { contrast: manual(20), look: "builtin:signature" } });
    expect(report.pasted[0]!.before).not.toHaveProperty("crop");
    const two = (await api.getSettings(2)).settings;
    expect(two).toMatchObject({ contrast: manual(20), orientation: 180, white_pick: null });
    expect(two.crop).not.toBeNull();
    expect((await api.getSettings(3)).settings.contrast).toEqual(manual(20));
    expect((await api.getSettings(4)).settings.contrast).toEqual({ kind: "auto" });
    await Promise.resolve();
    expect(events.sort()).toEqual([2, 3]);
    // The same paste again changes nothing: empty steps, no event.
    events.length = 0;
    const again = await api.pasteSettings([2], patch);
    expect(again.pasted).toEqual([{ id: 2, before: {}, after: {} }]);
    await Promise.resolve();
    expect(events).toEqual([]);
  });

  it("skips photos that are still developing or failed, and refuses a look that is gone", async () => {
    api.reset({ skipped: { DSC04203: "This file is not a raw file." } });
    await api.openPaths(ARW);
    const early = await api.pasteSettings([1, 3], { contrast: manual(10) });
    expect(early).toEqual({ pasted: [], skipped: [1, 3] });
    api.flush();
    expect((await api.pasteSettings([1, 3], { contrast: manual(10) })).skipped).toEqual([3]);
    const error = await rejection(api.pasteSettings([1], { look: "arri:3000:nothing", contrast: manual(50) }));
    expect(error.scope).toBe("looks");
    expect((await api.getSettings(1)).settings.contrast).toEqual(manual(10));
  });

  it("remembers the last menu state it was given", async () => {
    expect(api.menuState).toBeNull();
    await api.setMenuState({ copy_settings: true, paste_settings: false, paste_settings_to_picked: true });
    expect(api.menuState).toEqual({ copy_settings: true, paste_settings: false, paste_settings_to_picked: true });
    api.reset();
    expect(api.menuState).toBeNull();
  });
});

describe("Apple Auto (docs/19)", () => {
  it("is not available by default, as in the browser build: every photo is off and Auto is the rule", async () => {
    expect(await api.appleAutoStatus()).toEqual({
      available: false,
      reason: null,
      info: null,
      preference: true,
      mode: "from_preference",
      active: false,
      prompt: 3,
      table: 2,
      instructions: "default",
      instructions_sha256: "932b96907b63",
      own_instructions: "",
      default_instructions: MOCK_DEFAULT_INSTRUCTIONS,
      instructions_max_chars: 2000,
    });
    await api.openPaths(ARW);
    const rs = await api.getSettings(1);
    expect(rs.estimates).toEqual(MOCK_ESTIMATES);
    expect(rs.estimates.apple).toEqual({ state: "off" });
    expect(rs.settings.contrast).toEqual({ kind: "auto" });
  });

  it("says why it is not available on a Mac where Apple Intelligence is off", async () => {
    api.reset({ appleAuto: "unavailable" });
    expect(await api.appleAutoStatus()).toMatchObject({ available: false, reason: "apple_intelligence_off", active: false });
    expect(mockEstimates("unavailable", true).apple).toEqual({ state: "unavailable", reason: "apple_intelligence_off" });
  });

  it("answers through the table of section 5: +0.3 EV, 10 mired warmer, contrast +15, saturation −15, and the rest", () => {
    const e = mockEstimates("answered", true);
    expect([e.exposure_ev, e.kelvin, e.contrast, e.saturation, e.source]).toEqual([0.6, 5750, 15, -15, "apple"]);
    // Highlights −20, black −15, the shadows and the tint left empty (nothing set), teal shadows and warm highlights.
    expect([e.highlights, e.shadows, e.black, e.tint]).toEqual([-20, 0, -15, 0]);
    expect(e.balance_shadows[0]).toBeCloseTo(-0.866 * 0.55, 12);
    expect(e.balance_shadows[1]).toBeCloseTo(-0.5 * 0.55, 12);
    expect(e.balance_highlights[0]).toBeCloseTo(0.866 * 0.3, 12);
    expect(e.apple_steps).toEqual(MOCK_APPLE_ANSWER.steps);
    expect(e.apple).toMatchObject({ state: "answered", info: MOCK_APPLE_INFO });
    // Pending and failed are the rule, as is the preference off.
    for (const [mode, pref] of [["pending", true], ["failed", true], ["answered", false]] as const) {
      const rule = mockEstimates(mode, pref);
      expect([rule.exposure_ev, rule.kelvin, rule.contrast, rule.saturation, rule.source]).toEqual([0.3, 5450, 0, 0, "rule"]);
    }
    expect(mockEstimates("answered", false).apple).toEqual({ state: "off" });
    expect(mockEstimates("failed", true).apple).toMatchObject({ state: "failed", kind: "timeout" });
  });

  it("takes instructions of at most 2000 characters, saves the default as empty, and the answers say which were read", async () => {
    api.reset({ appleAuto: "answered" });
    await api.openPaths(ARW);
    api.flush();
    await expect(api.preferencesSet({ apple_auto_instructions: "x".repeat(2001) })).rejects.toMatchObject({ scope: "preferences" });
    expect((await api.preferencesGet()).apple_auto_instructions).toBe("");
    const changed: number[] = [];
    api.on("settings_changed", (e) => changed.push(e.photo_id));
    await api.preferencesSet({ apple_auto_instructions: "Moody night grade." });
    await vi.advanceTimersByTimeAsync(1);
    expect(changed).toHaveLength(12);
    const status = await api.appleAutoStatus();
    expect(status).toMatchObject({ instructions: "edited", own_instructions: "Moody night grade." });
    expect(status.instructions_sha256).toMatch(/^[0-9a-f]{12}$/);
    expect(status.instructions_sha256).not.toBe("932b96907b63");
    const rs = await api.getSettings(1);
    expect(rs.estimates.apple).toMatchObject({ state: "answered", info: { instructions: "edited" } });
    const saved = await api.preferencesSet({ apple_auto_instructions: MOCK_DEFAULT_INSTRUCTIONS });
    expect(saved.apple_auto_instructions).toBe("");
    expect(await api.appleAutoStatus()).toMatchObject({ instructions: "default", instructions_sha256: "932b96907b63" });
  });

  it("follows the preference at once and tells about every developed photo with settings_changed", async () => {
    api.reset({ appleAuto: "answered" });
    await api.openPaths(ARW);
    api.flush();
    expect(await api.appleAutoStatus()).toMatchObject({ available: true, active: true, info: MOCK_APPLE_INFO });
    expect((await api.getSettings(1)).values.contrast).toBe(15);
    const changed: number[] = [];
    api.on("settings_changed", (e) => changed.push(e.photo_id));
    await api.preferencesSet({ apple_auto: false });
    await vi.advanceTimersByTimeAsync(1);
    expect(changed).toHaveLength(12);
    expect((await api.getSettings(1)).values.contrast).toBe(0);
    expect(await api.appleAutoStatus()).toMatchObject({ available: true, active: false, preference: false });
    // A change that moves nothing says nothing.
    changed.length = 0;
    await api.preferencesSet({ remove_location_from_exports: true });
    await vi.advanceTimersByTimeAsync(1);
    expect(changed).toEqual([]);
  });

  it("Re-run puts contrast and saturation back to Auto only while Apple Auto acts on the photo", async () => {
    const cases = [
      ["pending", true, true],
      ["answered", true, true],
      ["failed", true, true],
      ["unavailable", true, false],
      ["answered", false, false],
      ["none", true, false],
    ] as const;
    for (const [appleAuto, preference, resets] of cases) {
      api.reset({ appleAuto });
      await api.openPaths(ARW);
      api.flush();
      await api.preferencesSet({ apple_auto: preference });
      await api.updateSettings(1, { exposure: { kind: "manual", ev: 1 }, contrast: manual(20), saturation: manual(-10) });
      const r = await api.rerunAuto(1);
      const what = `${appleAuto}, preference ${preference ? "on" : "off"}`;
      expect(r.auto.exposure, what).toBe(true);
      expect([r.auto.contrast, r.auto.saturation], what).toEqual([resets, resets]);
      if (!resets) expect([r.values.contrast, r.values.saturation], what).toEqual([20, -10]);
    }
  });

  it("setAppleAuto plays the answers arriving", async () => {
    api.reset({ appleAuto: "pending" });
    await api.openPaths(ARW);
    api.flush();
    expect((await api.getSettings(2)).estimates.apple).toEqual({ state: "pending" });
    const changed: number[] = [];
    api.on("settings_changed", (e) => changed.push(e.photo_id));
    api.setAppleAuto("answered");
    await vi.advanceTimersByTimeAsync(1);
    expect(changed).toHaveLength(12);
    expect((await api.getSettings(2)).estimates.apple.state).toBe("answered");
  });

  it("reads contrast and saturation as the core does: a number is Manual, 0 is Auto, a manual value is clamped", () => {
    expect(percentSetting(0)).toEqual({ kind: "auto" });
    expect(percentSetting(12)).toEqual({ kind: "manual", value: 12 });
    expect(percentSetting({ kind: "manual", value: -400 })).toEqual({ kind: "manual", value: -100 });
    expect(percentSetting({ kind: "auto" })).toEqual({ kind: "auto" });
    const rs = resolveSettings(1, { ...defaultSettings(), saturation: { kind: "manual", value: 9 } }, mockEstimates("answered", true));
    expect([rs.values.contrast, rs.values.saturation, rs.auto.contrast, rs.auto.saturation]).toEqual([15, 9, true, false]);
  });

  it("is set from the page address", () => {
    expect(optionsFromLocation("?mockAppleAuto=answered").appleAuto).toBe("answered");
    expect(optionsFromLocation("?mockAppleAuto=pending").appleAuto).toBe("pending");
    expect(optionsFromLocation("?mockAppleAuto=nonsense").appleAuto).toBeUndefined();
  });
});
