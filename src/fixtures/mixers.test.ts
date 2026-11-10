/**
 * The mock's approximation of `docs/25` §3 on the flat scenes (`fixtures/mixers.ts`): the Parallel's sum from its base, the Layer's modes and
 * opacity, the Splitter and the Combiner, bypass, and a graph without a live mixer being left to the plain path.
 */
import { describe, expect, it } from "vitest";
import type { DevelopSettings, NodeId, NodeLink, NodePatch, Port } from "../api/types";
import { sortedLinks } from "../state/graphOps";
import { applyPatch, defaultSettings } from "../state/settingsGraph";
import { blend, hasMixers, outputFeeder, sceneAt } from "./mixers";
import type { Scene } from "./scenes";

const link = (from: NodeId, to: NodeId, toPort: Port = "picture", fromPort: Port = "picture"): NodeLink => ({ from: { node: from, port: fromPort }, to: { node: to, port: toPort } });
const base: Scene = { name: "T", sky: "#404040", far: "#808080", near: "#202020", sun: "#ff8000", sx: 0, sy: 0 };
const brighten = (scene: Scene, by: number): Scene => {
  const plus = (hex: string): string => `#${[1, 3, 5].map((i) => Math.min(255, parseInt(hex.slice(i, i + 2), 16) + by).toString(16).padStart(2, "0")).join("")}`;
  return { ...scene, sky: plus(scene.sky), far: plus(scene.far), near: plus(scene.near), sun: plus(scene.sun) };
};
/** Every corrector adds its own number of levels: node 2 adds 10, node 3 adds 20. */
const apply = (item: { id: NodeId }, scene: Scene): Scene => (item.id === 1 ? scene : brighten(scene, (item.id as number) * 10 - 10));

function build(nodes: Record<string, NodePatch>, links: NodeLink[]): DevelopSettings {
  return applyPatch(defaultSettings(), { nodes, links: sortedLinks(links) });
}
const parallel = (bypass = false): DevelopSettings =>
  build(
    { "2": { kind: "corrector", pos: [1, 0] }, "3": { kind: "corrector", pos: [1, 1] }, "4": { kind: "parallel", pos: [2, 0], ...(bypass ? { bypass: true } : {}) } },
    [link("input", 1), link(1, 2), link(1, 3), link(2, 4, "picture.1"), link(3, 4, "picture.2"), link(4, "output")],
  );

describe("the Parallel", () => {
  it("adds each branch's change to the base: 10 and 20 levels over Main give 30", () => {
    const s = parallel();
    const out = sceneAt(s, base, outputFeeder(s), apply);
    expect(out.sky).toBe("#5e5e5e");
    expect(out.sun).toBe("#ff9e1e");
  });

  it("gives its base when bypassed", () => {
    const s = parallel(true);
    expect(sceneAt(s, base, outputFeeder(s), apply)).toEqual(base);
  });
});

describe("the Layer", () => {
  const layer = (mode: string, opacity: number, bypass = false): DevelopSettings =>
    build(
      { "2": { kind: "corrector", pos: [1, 0] }, "3": { kind: "layer", pos: [2, 0], mode, opacity, ...(bypass ? { bypass: true } : {}) } as NodePatch },
      [link("input", 1), link(1, 2), link(1, 3, "picture.1"), link(2, 3, "picture.2"), link(3, "output")],
    );
  it("Normal at 100 % shows the top layer, and at 0 % the bottom", () => {
    const top = sceneAt(layer("normal", 1), base, { node: 3, port: "picture" }, apply);
    expect(top).toEqual(brighten(base, 10));
    expect(sceneAt(layer("normal", 0), base, { node: 3, port: "picture" }, apply)).toEqual(base);
  });
  it("blends in linear light by the seven modes", () => {
    expect(blend("normal", 0.18, 0.09)).toBe(0.09);
    expect(blend("add", 0.18, 0.09)).toBeCloseTo(0.27, 12);
    expect(blend("subtract", 0.18, 0.09)).toBeCloseTo(0.09, 12);
    expect(blend("multiply", 0.18, 0.09)).toBeCloseTo(0.0162, 12);
    expect(blend("lighten", 0.18, 0.09)).toBe(0.18);
    expect(blend("darken", 0.18, 0.09)).toBe(0.09);
    expect(blend("difference", 0.09, 0.18)).toBeCloseTo(0.09, 12);
    const added = sceneAt(layer("add", 1), base, { node: 3, port: "picture" }, apply);
    expect(parseInt(added.far.slice(1, 3), 16)).toBeGreaterThan(parseInt(base.far.slice(1, 3), 16));
  });
  it("gives input 1 when bypassed", () => {
    expect(sceneAt(layer("add", 1, true), base, { node: 3, port: "picture" }, apply)).toEqual(base);
  });
});

describe("the Splitter and the Combiner", () => {
  const trio = (): DevelopSettings =>
    build(
      { "2": { kind: "splitter", pos: [1, 0] }, "3": { kind: "corrector", name: "Red", pos: [2, 0] }, "4": { kind: "combiner", pos: [3, 0] } },
      [link("input", 1), link(1, 2), link(2, 3, "picture", "red"), link(3, 4, "red"), link(2, 4, "green", "green"), link(2, 4, "blue", "blue"), link(4, "output")],
    );
  it("sends a grey picture of one channel to each output and joins three back", () => {
    const s = trio();
    const grey = sceneAt(s, base, { node: 2, port: "red" }, apply);
    expect(grey.sun).toBe("#ffffff");
    // Red lifted by 20 (node 3 adds 20), green and blue untouched: the sun's red goes to 255, its green and blue stay.
    const out = sceneAt(s, base, outputFeeder(s), apply);
    expect(out.sky).toBe("#544040");
    expect(out.sun).toBe("#ff8000");
  });
  it("is identity with the correctors at their identity", () => {
    const s = trio();
    expect(sceneAt(s, base, outputFeeder(s), (_i, scene) => scene)).toEqual(base);
  });
});

describe("hasMixers", () => {
  it("is true for a live mixer and false for a dead one or none", () => {
    expect(hasMixers(parallel())).toBe(true);
    expect(hasMixers(defaultSettings())).toBe(false);
    const dead = build({ "2": { kind: "parallel", pos: [1, 0] } }, [link("input", 1), link(1, 2, "picture.1"), link(1, "output")]);
    expect(hasMixers(dead)).toBe(false);
  });
});
