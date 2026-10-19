/**
 * The colour warp on the mock API (`docs/02` C9b, `docs/11` §Colour warp): the setting, its clamping, the scatter of the photo's
 * colours, Pick colour's sample, copy and paste, and a picture that follows the pins.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clampSettings, copiedPatch, createMockApi, defaultSettings, MOCK_ESTIMATES, resolveSettings } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { ColorWarpPin, ViewportFrame } from "./api/types";
import { lockedPin, movingPin, WARP_COORD_LIMIT, WARP_MAX_PINS } from "./components/warpMath";
import { adjustScene, developScene } from "./fixtures/grade";
import { SCENES } from "./fixtures/scenes";
import { hexChroma, hexToLogC4, logC4ToHex, sceneScatter, warpScene } from "./fixtures/warp";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW"];
/** DSC04204, the mockup's current photo: sky #7FA6C7, far #E6ECF2, near #3F5468, sun #FFF6E0. */
const SCENE = SCENES[3]!;
const NEUTRAL_ADJUSTMENT = { ev: 0.3, autoEv: 0.3, kelvin: 5450, autoKelvin: 5450, tint: 0, autoTint: 0 };

let api: MockApi;
beforeEach(() => {
  vi.useFakeTimers();
  api = createMockApi();
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

describe("the setting", () => {
  it("is no pins by default, and resolves as it is stored", () => {
    expect(defaultSettings().color_warp).toEqual({ pins: [] });
    const settings = { ...defaultSettings(), color_warp: { pins: [movingPin([0.1, 0], [0.1, 0.05])] } };
    expect(resolveSettings(1, settings, MOCK_ESTIMATES).values.color_warp).toEqual(settings.color_warp);
  });

  it("is clamped as the core clamps it: sixteen pins, every value in range", () => {
    const wild: ColorWarpPin = { source: [3, 4], target: [0.1, 0.1], locked: false, range: 1, shadows: 2, highlights: -1, pivot_ev: 9, exposure_ev: 5 };
    const pins = [wild, ...Array.from({ length: WARP_MAX_PINS + 3 }, () => lockedPin([0.1, 0]))];
    const out = clampSettings({ ...defaultSettings(), color_warp: { pins } }).color_warp.pins;
    expect(out).toHaveLength(WARP_MAX_PINS);
    expect(Math.hypot(...out[0]!.source)).toBeCloseTo(WARP_COORD_LIMIT, 12);
    expect(out[0]).toMatchObject({ range: 0.25, shadows: 1, highlights: 0, pivot_ev: 6, exposure_ev: 2 });
    // Settings from before the warp have none.
    const { color_warp: _gone, ...older } = defaultSettings();
    expect(clampSettings(older as never).color_warp).toEqual({ pins: [] });
  });

  it("a patch replaces the whole list", async () => {
    await api.openPaths(ARW);
    await api.updateSettings(1, { color_warp: { pins: [movingPin([0.1, 0]), lockedPin([0, 0.1])] } });
    const rs = await api.updateSettings(1, { color_warp: { pins: [movingPin([-0.1, 0], [-0.1, 0.05])] } });
    expect(rs.settings.color_warp.pins).toEqual([movingPin([-0.1, 0], [-0.1, 0.05])]);
  });

  it("is copied with the rest of the grade and pasted onto other photos as one step", async () => {
    await api.openPaths(ARW);
    api.flush();
    const pins = [movingPin([0.1, 0], [0.1, 0.05]), lockedPin([-0.1, 0])];
    await api.updateSettings(1, { color_warp: { pins } });
    const patch = await api.copySettings(1);
    expect(patch.color_warp).toEqual({ pins });
    expect(copiedPatch(await api.getSettings(1)).color_warp).toEqual({ pins });
    const report = await api.pasteSettings([2, 3], patch);
    expect(report.pasted.map((p) => p.after.color_warp)).toEqual([{ pins }, { pins }]);
    expect(report.pasted[0]?.before.color_warp).toEqual({ pins: [] });
    expect((await api.getSettings(3)).settings.color_warp.pins).toEqual(pins);
  });
});

describe("chroma_scatter", () => {
  beforeEach(async () => {
    await api.openPaths(ARW);
  });

  it("gives at most 2048 points of the photo's colours, the same on every call", async () => {
    const { points } = await api.chromaScatter(4);
    expect(points.length % 2).toBe(0);
    expect(points.length / 2).toBeLessThanOrEqual(2048);
    expect(points.length / 2).toBeGreaterThan(1000);
    expect((await api.chromaScatter(4)).points).toEqual(points);
    // Most of the frame is sky and ground, both blue: the mean sits on the blue side of the plane.
    const mean = [0, 1].map((k) => points.filter((_, i) => i % 2 === k).reduce((a, b) => a + b, 0) / (points.length / 2));
    expect(mean[0]).toBeLessThan(-0.03);
    expect(mean[1]).toBeLessThan(-0.03);
  });

  it("reads the colours at the warp's input: the white balance moves them, the pins do not", async () => {
    const before = (await api.chromaScatter(4)).points;
    await api.updateSettings(4, { color_warp: { pins: [movingPin(hexChroma(SCENE.sky), [0.1, 0.1])] } });
    expect((await api.chromaScatter(4)).points).toEqual(before);
    await api.updateSettings(4, { white_balance: { kind: "manual", kelvin: 9000, tint: 0 } });
    expect((await api.chromaScatter(4)).points).not.toEqual(before);
  });
});

describe("sample_chroma", () => {
  beforeEach(async () => {
    await api.openPaths(ARW);
    await api.setCurrent(4);
  });

  it("is the chroma of the colour under a point of the cell, the one the scatter is made of", async () => {
    await api.viewportSetRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
    const sky = await api.sampleChroma(20, 20);
    const near = await api.sampleChroma(750, 950);
    expect([sky?.a, sky?.b]).toEqual(hexChroma(SCENE.sky));
    expect([near?.a, near?.b]).toEqual(hexChroma(SCENE.near));
  });

  it("is null on the paper around a letterboxed photo", async () => {
    await api.updateSettings(4, { orientation: 90 });
    await api.viewportSetRect({ x: 0, y: 0, width: 900, height: 600, scale: 2 });
    expect(await api.sampleChroma(20, 300)).toBeNull();
    expect(await api.sampleChroma(450, 300)).not.toBeNull();
  });
});

describe("the picture follows the pins", () => {
  it("round-trips a colour exactly when nothing moves it, and a pin moves the colour it was made from", () => {
    for (const hex of [SCENE.sky, SCENE.far, SCENE.near, SCENE.sun, "#C99A63"]) expect(logC4ToHex(hexToLogC4(hex))).toBe(hex);
    expect(warpScene(SCENE, [])).toBe(SCENE);
    expect(warpScene(SCENE, [lockedPin(hexChroma(SCENE.sky))])).toBe(SCENE);
    const moved = warpScene(SCENE, [movingPin(hexChroma(SCENE.sky), [0.1, 0.1])]);
    expect(moved.sky).not.toBe(SCENE.sky);
    // The sky goes from blue toward orange: more red, less blue.
    const rgb = (hex: string) => [1, 3, 5].map((i) => Number.parseInt(hex.slice(i, i + 2), 16));
    expect(rgb(moved.sky)[0]).toBeGreaterThan(rgb(SCENE.sky)[0]!);
    expect(rgb(moved.sky)[2]).toBeLessThan(rgb(SCENE.sky)[2]!);
    // The pale far hill lies on the move's way (a quarter of a range from its line), so the flow sweeps it along, unless a lock holds it.
    expect(moved.far).not.toBe(SCENE.far);
    const held = warpScene(SCENE, [movingPin(hexChroma(SCENE.sky), [0.1, 0.1]), lockedPin(hexChroma(SCENE.far))]);
    expect(held.far).toBe(SCENE.far);
    expect(held.sky).not.toBe(SCENE.sky);
  });

  it("moves the scene after the light changes and before the look", () => {
    const pins = [movingPin(hexChroma(SCENE.sky), [0.1, 0.1])];
    const adjusted = adjustScene(SCENE, { ...NEUTRAL_ADJUSTMENT, warp: pins });
    expect(adjusted.sky).toBe(warpScene(SCENE, pins).sky);
    const developed = developScene(SCENE, { slug: "neutral", intensity: 100 }, { ...NEUTRAL_ADJUSTMENT, warp: pins });
    expect(developed.sky).toBe(adjusted.sky);
  });

  it("is what the fallback viewport's frame shows", async () => {
    api = createMockApi({ viewport: "fallback" });
    const frames: ViewportFrame[] = [];
    api.on("viewport_frame", (f) => frames.push(f));
    await api.openPaths(ARW);
    api.flush();
    await api.setCurrent(4);
    await api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    const pins = [movingPin(hexChroma(SCENE.sky), [0.1, 0.1])];
    await api.updateSettings(4, { color_warp: { pins } });
    await Promise.resolve();
    const svg = decodeURIComponent(frames.at(-1)!.url);
    expect(svg).toContain(warpScene(SCENE, pins).sky);
    expect(svg).not.toContain(SCENE.sky);
  });

  it("scatters a flat scene deterministically, seeded", () => {
    expect(sceneScatter(SCENE, 1)).toEqual(sceneScatter(SCENE, 1));
    expect(sceneScatter(SCENE, 1)).not.toEqual(sceneScatter(SCENE, 2));
  });
});
