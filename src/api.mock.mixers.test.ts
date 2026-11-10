/**
 * The panel signals at a mixer, a Splitter or a Combiner on the mock API (`docs/25` §5.7): the histogram, the scatter and Pick colour read
 * the picture after the node; at a Splitter, the picture on the link into it. The core's words for a node the photo does not have stay.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { AppError, NodeLink, SettingsPatch } from "./api/types";

const ARW = ["~/Pictures/Lisbon/DSC04201.ARW", "~/Pictures/Lisbon/DSC04202.ARW"];
const link = (from: number | "input", fromPort: string, to: number | "output", toPort = "picture"): NodeLink =>
  ({ from: { node: from, port: fromPort }, to: { node: to, port: toPort } }) as NodeLink;

let api: MockApi;
beforeEach(async () => {
  vi.useFakeTimers();
  api = createMockApi();
  await api.openPaths(ARW);
  api.flush();
  await api.viewportSetRect({ x: 0, y: 0, width: 1500, height: 1000, scale: 1 });
  await api.setCurrent(1);
});
afterEach(() => {
  api.reset();
  vi.useRealTimers();
});

const rejection = async (promise: Promise<unknown>): Promise<AppError> => {
  try {
    await promise;
  } catch (error) {
    return error as AppError;
  }
  throw new Error("expected a rejection");
};

/** Input → Main → 2 (a cooler trim) → Layer 3 (one input) → Output. */
const layerOfNode2: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Cool", trim: { exposure_ev: 1.5, temperature: -40 }, pos: [1, 0] },
    "3": { kind: "layer", pos: [2, 0] },
  },
  links: [link("input", "picture", 1), link(1, "picture", 2), link(2, "picture", 3, "picture.1"), link(3, "picture", "output")],
};

/** Input → Main → 2 (a cooler trim) → Splitter 3 → Combiner 4 (the three channels straight back) → Output. */
const trio: SettingsPatch = {
  nodes: {
    "2": { kind: "corrector", name: "Cool", trim: { exposure_ev: 1.5, temperature: -40 }, pos: [1, 0] },
    "3": { kind: "splitter", pos: [2, 0] },
    "4": { kind: "combiner", pos: [3, 0] },
  },
  links: [
    link("input", "picture", 1),
    link(1, "picture", 2),
    link(2, "picture", 3),
    link(3, "red", 4, "red"),
    link(3, "green", 4, "green"),
    link(3, "blue", 4, "blue"),
    link(4, "picture", "output"),
  ],
};

describe("the panel signals at a mixer, a Splitter or a Combiner", () => {
  it("read a Layer of one input as that input's picture, by the histogram, the scatter and Pick colour", async () => {
    await api.updateSettings(1, layerOfNode2);
    const histogram = await api.histogram(1, 2);
    expect(await api.histogram(1, 3)).toEqual(histogram);
    expect(await api.chromaScatter(1, 3)).toEqual(await api.chromaScatter(1, 2));
    const pick = await api.sampleChroma(20, 20, 3);
    expect(pick).toEqual(await api.sampleChroma(20, 20, 2));
    expect(pick).not.toEqual(await api.sampleChroma(20, 20));
    // Either tap: the picture after the node.
    expect(await api.sampleChroma(20, 20, 3, "node_input")).toEqual(pick);
  });

  it("read a Splitter as its input picture and a Combiner as the picture it joins", async () => {
    await api.updateSettings(1, trio);
    const cooled = await api.histogram(1, 2);
    expect(await api.histogram(1, 3)).toEqual(cooled);
    expect(await api.chromaScatter(1, 3)).toEqual(await api.chromaScatter(1, 2));
    expect(await api.sampleChroma(20, 20, 3)).toEqual(await api.sampleChroma(20, 20, 2));
    // The channels go back where they came from: the Combiner's picture is node 2's.
    expect(await api.histogram(1, 4)).toEqual(cooled);
    expect(await api.sampleChroma(20, 20, 4)).toEqual(await api.sampleChroma(20, 20, 2));
  });

  it("read a Splitter straight after the Input as the Input's picture", async () => {
    await api.updateSettings(1, {
      nodes: { "2": { kind: "splitter", pos: [1, 1] } },
      links: [link("input", "picture", 1), link(1, "picture", "output"), link("input", "picture", 2)],
    });
    expect(await api.histogram(1, 2)).toEqual(await api.histogram(1, "input"));
  });

  it("refuse a Splitter with nothing linked in, and a node the photo does not have, in words", async () => {
    await api.updateSettings(1, { nodes: { "2": { kind: "splitter", pos: [1, 1] } }, links: [link("input", "picture", 1), link(1, "picture", "output")] });
    expect(await rejection(api.histogram(1, 2))).toMatchObject({ scope: "render", message: "Node 2 has no picture to read here." });
    expect(await rejection(api.histogram(1, 7))).toMatchObject({ scope: "render", message: "This photo has no node 7." });
  });
});
