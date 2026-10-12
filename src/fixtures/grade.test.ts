import { describe, expect, it } from "vitest";
import { adjustScene, developScene, flatScene, gradeScene, LOOK_GRADES, previewDataUrl, previewSvg } from "./grade";
import { SCENES, sceneSvg } from "./scenes";

const scene = SCENES[3]!;
const none = { ev: 0.3, autoEv: 0.3, kelvin: 5450, autoKelvin: 5450, tint: 0, autoTint: 0 };

describe("grading the flat scenes", () => {
  it("Neutral at any strength changes nothing", () => {
    expect(gradeScene(scene, LOOK_GRADES.neutral!, 1)).toEqual(scene);
  });

  it("a look at 0 % changes nothing, at 100 % changes the colours", () => {
    const teal = LOOK_GRADES["teal-and-orange"]!;
    expect(gradeScene(scene, teal, 0)).toEqual(scene);
    expect(gradeScene(scene, teal, 1).near).not.toBe(scene.near);
  });

  it("leaves the geometry alone", () => {
    const out = gradeScene(scene, LOOK_GRADES["night-blue"]!, 1);
    expect([out.name, out.sx, out.sy]).toEqual([scene.name, scene.sx, scene.sy]);
  });

  it("Monochrome desaturates fully", () => {
    const out = gradeScene(scene, LOOK_GRADES.monochrome!, 1);
    const [r, g, b] = [1, 3, 5].map((i) => Number.parseInt(out.sky.slice(i, i + 2), 16));
    expect(Math.abs(r! - g!)).toBeLessThan(8);
    expect(Math.abs(g! - b!)).toBeLessThan(8);
  });

  it("the Before view is flatter than the scene", () => {
    const flat = flatScene(scene);
    expect(flat.sky).not.toBe(scene.sky);
    expect(flat.name).toBe(scene.name);
  });

  it("does not move colours that are at the automatic values", () => {
    expect(adjustScene(scene, none)).toEqual(scene);
    expect(developScene(scene, { slug: "neutral", intensity: 100 }, none)).toEqual(scene);
  });

  it("warms for a higher kelvin and brightens for more exposure", () => {
    const warm = adjustScene(scene, { ...none, kelvin: 7000 });
    expect(warm.sky).not.toBe(scene.sky);
    const bright = adjustScene(scene, { ...none, ev: 1.3 });
    const lum = (hex: string): number => Number.parseInt(hex.slice(1, 3), 16);
    expect(lum(bright.sky)).toBeGreaterThan(lum(scene.sky));
    expect(lum(adjustScene(scene, { ...none, ev: -0.7 }).sky)).toBeLessThan(lum(scene.sky));
  });

  it("clamps the intensity and falls back to Neutral for an unknown look", () => {
    expect(developScene(scene, { slug: "no-such-look", intensity: 100 }, null)).toEqual(scene);
    expect(developScene(scene, { slug: "teal-and-orange", intensity: 500 }, null)).toEqual(developScene(scene, { slug: "teal-and-orange", intensity: 100 }, null));
  });
});

describe("previewSvg", () => {
  it("draws the same shapes as the fixture, as an SVG of the frame size", () => {
    const svg = previewSvg(scene);
    expect(svg).toContain('viewBox="0 0 1500 1000"');
    for (const line of sceneSvg(scene).split("\n").filter((l) => l.startsWith("<rect") || l.startsWith("<circle") || l.startsWith("<path"))) {
      expect(svg).toContain(line);
    }
  });

  it("turns the frame on its side for a quarter turn", () => {
    expect(previewSvg(scene, { orientation: 90 })).toContain('viewBox="0 0 1000 1500"');
    expect(previewSvg(scene, { orientation: 270 })).toContain('viewBox="0 0 1000 1500"');
    expect(previewSvg(scene, { orientation: 180 })).toContain('viewBox="0 0 1500 1000"');
  });

  it("shows only the crop", () => {
    const svg = previewSvg(scene, { crop: { x: 0.1, y: 0.2, width: 0.5, height: 0.5 } });
    expect(svg).toContain('width="750"');
    expect(svg).toContain('height="500"');
    expect(svg).toContain('viewBox="150 200 750 500"');
  });

  it("straightens about the middle", () => {
    expect(previewSvg(scene, { angle_deg: 3 })).toContain("rotate(3 750 500)");
    expect(previewSvg(scene)).not.toContain("rotate(0");
  });

  it("makes a data URL", () => {
    expect(previewDataUrl(scene)).toMatch(/^data:image\/svg\+xml;charset=utf-8,%3Csvg/);
  });
});
