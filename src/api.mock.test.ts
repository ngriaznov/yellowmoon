import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  clampSettings,
  createMockApi,
  defaultSettings,
  expandNameTemplate,
  lookSlug,
  MOCK_ESTIMATES,
  resolveSettings,
  type MockApi,
} from "./api.mock";
import type { AppError, ExportJob, ExportOptions, ExportProgress, PhotoProgress, ThumbnailReady } from "./api/types";

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

  it("adding photos to an open session changes nothing", async () => {
    const first = await api.openPaths(ARW);
    const again = await api.openPaths(ARW);
    expect(again.photos.map((p) => p.id)).toEqual(first.photos.map((p) => p.id));
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

  it("sends each thumbnail as a PNG of at most 240 px", async () => {
    const { thumbnails } = listen();
    await api.openPaths(ARW);
    await vi.advanceTimersByTimeAsync(4500);
    expect(thumbnails.map((t) => t.id)).toEqual(Array.from({ length: 12 }, (_, i) => i + 1));
    for (const { png_bytes } of thumbnails) {
      expect(png_bytes.slice(0, 8)).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
      const width = new DataView(new Uint8Array(png_bytes).buffer).getUint32(16);
      expect(width).toBeLessThanOrEqual(240);
    }
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
    expect(r.auto).toEqual({ exposure: true, light_source: true, white_balance: true, sharpen: true, noise_reduction: true });
    expect(r.values).toMatchObject({ exposure_ev: 0.3, kelvin: 5450, tint: 0 });
    expect(r.values.sharpen).toEqual({ amount: 120, radius_px: 0.8 });
    expect(r.values.noise_reduction).toEqual({ luminance: 30, color: 50, detail: 50 });
    expect(r.estimates).toEqual(MOCK_ESTIMATES);
    expect(r.settings.exposure).toEqual({ kind: "auto" });
    expect(r.settings).toEqual(defaultSettings());
  });

  it("turns a field Manual when patched, and keeps the other fields Auto", async () => {
    const r = await api.updateSettings(1, { exposure: { kind: "manual", ev: -0.5 }, contrast: 12 });
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
      contrast: -400,
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

  it("rerunAuto resets exposure, white balance and noise to Auto", async () => {
    await api.updateSettings(1, {
      exposure: { kind: "manual", ev: 1 },
      light_source: "tungsten",
      white_balance: { kind: "manual", kelvin: 3200, tint: 5 },
      noise_reduction: { kind: "manual", luminance: 80, color: 80, detail: 10 },
      contrast: 20,
    });
    const r = await api.rerunAuto(1);
    expect(r.auto).toEqual({ exposure: true, light_source: true, white_balance: true, sharpen: true, noise_reduction: true });
    expect(r.values.contrast).toBe(20); // other controls are not touched
  });

  it("emits render_stats after a change", async () => {
    const stats: unknown[] = [];
    api.on("render_stats", (e) => stats.push(e));
    await api.updateSettings(1, { contrast: 5 });
    await vi.advanceTimersByTimeAsync(1);
    expect(stats).toEqual([{ ms: 42, path: "neural", space: "display_p3" }]);
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

  it("imports the looks when they aren't there yet, and only once", async () => {
    api.reset({ looksImported: false });
    expect(await api.looksList()).toHaveLength(2);
    const events: unknown[] = [];
    api.on("looks_changed", (e) => events.push(e));
    const first = await api.looksImport("~/Downloads/looks.zip");
    expect(first.added).toHaveLength(7);
    expect(await api.looksList()).toHaveLength(9);
    await vi.advanceTimersByTimeAsync(1);
    expect(events).toEqual([null]);
    expect((await api.looksImport("~/Downloads/looks.zip")).added).toEqual([]);
    expect(await api.looksList()).toHaveLength(9);
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

  it("writes the equivalent command line, as the mockup does", async () => {
    expect(await api.cliPreview(job("picked"))).toBe(
      [
        "yellowmoon render DSC04203.ARW DSC04204.ARW DSC04205.ARW \\",
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
    expect(tiff).toContain("yellowmoon render ~/Pictures/Lisbon \\");
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
    expect(await api.installCliLink()).toEqual({ path: "/usr/local/bin/yellowmoon" });
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

  it("stores a white pick as a point in image coordinates", async () => {
    await api.viewportSetRect({ x: 0, y: 0, width: 1000, height: 500, scale: 2 });
    await api.viewportInput({ kind: "pick_white", x: 250, y: 100 });
    expect((await api.getSettings(1)).values.white_pick).toEqual([0.25, 0.2]);
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
      e.png_bytes.length = 0;
    });
    api.on("thumbnail_ready", (e) => seen.push(e));
    await api.openPaths(ARW);
    api.flush();
    expect(seen.filter((e) => e.png_bytes.length > 0)).toHaveLength(12);
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
