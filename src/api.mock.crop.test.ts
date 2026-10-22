/**
 * Auto crop on the mock API (`docs/20-auto-crop.md` §10): the fixed proposal (a 1.4° turn, a 90 % frame a little left of centre), the
 * whole photo kept for a "keep" photo, the refusal while a photo develops, and `Crop.auto` kept by the setting.
 */
import { afterEach, describe, expect, it } from "vitest";
import { createMockApi, mockCropSuggestion } from "./api.mock";
import type { MockApi, MockOptions } from "./api.mock";
import type { CropAspect, NormRect, Orientation } from "./api/types";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW"];

let api: MockApi | undefined;
afterEach(() => api?.reset());

async function open(options: MockOptions = {}): Promise<MockApi> {
  api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  await api.openPaths(ARW);
  api.flush();
  return api;
}

const inside = (r: NormRect): boolean => r.x >= 0 && r.y >= 0 && r.x + r.width <= 1 + 1e-9 && r.y + r.height <= 1 + 1e-9;

describe("suggest_crop on the mock", () => {
  it("levels by 1.4° and proposes a frame of 90 % a little left of centre, then the kept frame, all inside the canvas", async () => {
    const mock = await open();
    for (const [aspect, orientation] of [["original", 0], ["square", 0], ["four_by_five", 90], ["sixteen_by_nine", 270], ["free", 180]] as Array<[CropAspect, Orientation]>) {
      const s = await mock.suggestCrop(1, aspect, orientation);
      expect(s).toMatchObject({ straighten: "rotate", angle_deg: 1.4, keep: false, subject: "camera" });
      expect(s.frames).toHaveLength(2);
      const [best, kept] = [s.frames[0]!, s.frames[1]!];
      expect(best.rect.width).toBeCloseTo(0.9 * kept.rect.width, 5);
      expect(best.rect.height).toBeCloseTo(0.9 * kept.rect.height, 5);
      expect(best.rect.x + best.rect.width / 2).toBeLessThan(0.5);
      expect(kept.score).toBe(0);
      expect(s.frames.every((f) => inside(f.rect))).toBe(true);
    }
  });

  it("gives the frame the ratio of the aspect on the turned canvas: 4:5 follows the canvas, 16:9 too", async () => {
    const mock = await open();
    // 8640 × 5760, turned a quarter: a portrait canvas 2 wide for 3 high.
    const ratio = (rect: NormRect, canvas: number): number => (rect.width / rect.height) * canvas;
    expect(ratio((await mock.suggestCrop(1, "four_by_five", 0)).frames[1]!.rect, 1.5)).toBeCloseTo(5 / 4, 4);
    expect(ratio((await mock.suggestCrop(1, "four_by_five", 90)).frames[1]!.rect, 2 / 3)).toBeCloseTo(4 / 5, 4);
    expect(ratio((await mock.suggestCrop(1, "sixteen_by_nine", 90)).frames[1]!.rect, 2 / 3)).toBeCloseTo(9 / 16, 4);
    expect(ratio((await mock.suggestCrop(1, "square", 0)).frames[1]!.rect, 1.5)).toBeCloseTo(1, 4);
  });

  it("keeps a photo whose name has keep in it whole and level", async () => {
    const mock = await open({ photoOverrides: { DSC04202: { file_name: "DSC04202_keep.ARW" } } });
    const s = await mock.suggestCrop(2, "original", 0);
    expect(s).toEqual({ straighten: "level", angle_deg: 0, frames: [{ rect: { x: 0, y: 0, width: 1, height: 1 }, score: 0 }], keep: true, subject: "picture" });
    expect(mockCropSuggestion((await mock.sessionState())!.photos[1]!, "original", 0)).toEqual(s);
  });

  it("refuses while the photo is still developing", async () => {
    api = createMockApi({ speed: Number.POSITIVE_INFINITY, holdAt: 0 });
    await api.openPaths(ARW);
    await expect(api.suggestCrop(1, "original", 0)).rejects.toMatchObject({ scope: "open" });
  });

  it("keeps Crop.auto in the setting, and a crop without it reads as not Auto", async () => {
    const mock = await open();
    const rect = { x: 0.05, y: 0.05, width: 0.8, height: 0.8 };
    let rs = await mock.updateSettings(1, { crop: { rect, angle_deg: 1.4, aspect: "original", auto: true } });
    expect(rs.settings.crop?.auto).toBe(true);
    rs = await mock.updateSettings(1, { crop: { rect, angle_deg: 1.4, aspect: "original" } });
    expect(rs.settings.crop?.auto ?? false).toBe(false);
  });
});
