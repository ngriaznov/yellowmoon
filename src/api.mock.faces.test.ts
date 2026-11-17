/**
 * What the mock says for the Face Refinement node's faces (`docs/27` decision 9 and S-11b): the core finds the faces of the photo being looked at in the
 * background when a node exists on it, `viewport_busy` is `finding` meanwhile, `face_state` says `finding` and then `ready` with the numbered faces (or `failed`),
 * and `face_state(photo_id)` answers the same. `?mockFaces=` picks the outcome.
 */
import { afterEach, describe, expect, it } from "vitest";
import { createMockApi, FACES_FAILED, FACES_MODELS_MISSING, MOCK_FACES, optionsFromLocation } from "./api.mock";
import type { MockApi, MockOptions } from "./api.mock";
import type { FaceState, SettingsPatch } from "./api/types";

let api: MockApi | undefined;
afterEach(() => {
  api?.reset();
  api = undefined;
});

const link = (from: "input" | number, to: "output" | number) => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const NODE: SettingsPatch = {
  nodes: { "2": { kind: "face_refinement", name: "Face Refinement", pos: [1, 0] } },
  links: [link("input", 1), link(1, 2), link(2, "output")],
};
const tick = (ms = 20): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** DSC04204, open and developed, with a 600 × 400 cell, and what the mock says while it runs. */
async function open(options: MockOptions = {}, settings: SettingsPatch = {}) {
  const a = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options, initialSettings: { DSC04204: settings } });
  api = a;
  const seen = { busy: [] as Array<{ busy: boolean; cpu: boolean; finding: boolean }>, states: [] as FaceState[] };
  a.on("viewport_busy", (e) => seen.busy.push(e));
  a.on("face_state", (e) => seen.states.push(e));
  await a.openPaths(["/photos"]);
  a.flush();
  const id = (await a.sessionState()).photos[3]!.id;
  await a.setCurrent(id);
  await a.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  return { a, id, seen };
}

describe("face_state (docs/27 S-11b)", () => {
  it("finds nothing for a photo with no Face Refinement node, and answers finding", async () => {
    const { a, id, seen } = await open();
    await tick();
    expect(seen.states).toEqual([]);
    expect(await a.faceState(id)).toEqual({ photo_id: id, phase: "finding", faces: [], message: null });
  });

  it("finds the faces in the background when the node appears: finding, then ready with two faces numbered from left to right, and not again", async () => {
    const { a, id, seen } = await open();
    await a.updateSettings(id, NODE);
    await tick();
    expect(seen.busy).toEqual([
      { busy: true, cpu: false, finding: true },
      { busy: false, cpu: false, finding: false },
    ]);
    expect(seen.states).toEqual([
      { photo_id: id, phase: "finding", faces: [], message: null },
      { photo_id: id, phase: "ready", faces: [...MOCK_FACES], message: null },
    ]);
    expect(MOCK_FACES.map((f) => f.number)).toEqual([1, 2]);
    expect(MOCK_FACES[0]!.x).toBeLessThan(MOCK_FACES[1]!.x);
    expect(await a.faceState(id)).toEqual({ photo_id: id, phase: "ready", faces: [...MOCK_FACES], message: null });
    // A slider is no new analysis.
    await a.updateSettings(id, { nodes: { "2": { all: { blemish: 80 } } } });
    await tick();
    expect(seen.states).toHaveLength(2);
  });

  it("?mockFaces=none is ready with no face, fail fails the first time with the core's sentence, missing is models_missing and runs nothing", async () => {
    const none = await open({ faces: "none" });
    await none.a.updateSettings(none.id, NODE);
    await tick();
    expect(await none.a.faceState(none.id)).toEqual({ photo_id: none.id, phase: "ready", faces: [], message: null });
    api?.reset();
    const fail = await open({ faces: "fail" });
    await fail.a.updateSettings(fail.id, NODE);
    await tick();
    expect(fail.seen.states.at(-1)).toEqual({ photo_id: fail.id, phase: "failed", faces: [], message: FACES_FAILED });
    api?.reset();
    const missing = await open({ faces: "missing" });
    await missing.a.updateSettings(missing.id, NODE);
    await tick();
    expect(missing.seen.states).toEqual([]);
    expect(missing.seen.busy.filter((e) => e.finding)).toEqual([]);
    expect(await missing.a.faceState(missing.id)).toEqual({ photo_id: missing.id, phase: "models_missing", faces: [], message: FACES_MODELS_MISSING });
  });

  it("reads ?mockFaces= from the page URL", () => {
    expect(optionsFromLocation("").faces).toBeUndefined();
    expect(optionsFromLocation("?mockFaces=none").faces).toBe("none");
    expect(optionsFromLocation("?mockFaces=missing").faces).toBe("missing");
    expect(optionsFromLocation("?mockFaces=bogus").faces).toBeUndefined();
  });
});

describe("face_crop (docs/27 S-11c)", () => {
  it("answers with a picture for each face found, and says so in a plain sentence before and past the end", async () => {
    const { a, id } = await open();
    await expect(a.faceCrop(id, 1)).rejects.toMatchObject({ scope: "render", message: "The faces are not found yet. Try again in a moment." });
    await a.updateSettings(id, NODE);
    await tick();
    const one = await a.faceCrop(id, 1);
    const two = await a.faceCrop(id, 2);
    expect(one).toMatch(/^data:image\/svg\+xml/);
    expect(one).not.toBe(two);
    await expect(a.faceCrop(id, 3)).rejects.toMatchObject({ scope: "render", message: "This photo has no face 3. It has 2." });
  });
});
