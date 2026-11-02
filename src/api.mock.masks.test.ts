/**
 * What the mock says for the Mask tab (`docs/23` §9): `mask_kinds` from `?mockMasks=` (every kind by default), `viewport_map_points`
 * between the photo frame and the cell as the core gives them (without the straightening and the zoom, which the mock's flat picture has not), and
 * the copy rule for the rows.
 */
import { afterEach, describe, expect, it } from "vitest";
import { ALL_MASK_KINDS, BUILT_MASK_KINDS, createMockApi, optionsFromLocation, parseMockMasks } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { MaskItem, SettingsPatch } from "./api/types";
import { copiedPatch } from "./fixtures/graph";
import { newMask } from "./state/maskRows";

let api: MockApi | undefined;
afterEach(() => {
  api?.reset();
  api = undefined;
});

/** DSC04204 (6000 × 4000), open and developed, with a 600 × 400 cell. */
async function open(settings: SettingsPatch = {}): Promise<MockApi> {
  api = createMockApi({ speed: Number.POSITIVE_INFINITY, initialSettings: { DSC04204: settings } });
  await api.openPaths(["/photos"]);
  api.flush();
  await api.setCurrent((await api.sessionState()).photos[3]!.id);
  await api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  return api;
}

describe("mask_kinds (the Add mask menu's list)", () => {
  it("is every kind (Object, Subject and People first, N-18 and N-19), unless the page URL says otherwise", async () => {
    const a = createMockApi({ speed: Number.POSITIVE_INFINITY });
    api = a;
    expect(await a.maskKinds()).toEqual(["object", "subject", "people", "background", "linear", "radial", "brush", "color_range", "brightness_range"]);
    expect(await a.maskKinds()).toEqual([...BUILT_MASK_KINDS]);
  });

  it("is what ?mockMasks= names: all, noai, nomac, built, or a list of kinds, in the menu's order", () => {
    expect(optionsFromLocation("").maskKinds).toBeUndefined();
    expect(optionsFromLocation("?mockMasks=all").maskKinds).toEqual([...ALL_MASK_KINDS]);
    // No AI: Object, Subject and People are left out; Background needs no model download and no Mac, the blur map comes from the photo.
    expect(optionsFromLocation("?mockMasks=noai").maskKinds).toEqual(["background", "linear", "radial", "brush", "color_range", "brightness_range"]);
    // No Mac: Subject and People are left out.
    expect(optionsFromLocation("?mockMasks=nomac").maskKinds).toEqual(["object", "background", "linear", "radial", "brush", "color_range", "brightness_range"]);
    expect(optionsFromLocation("?mockMasks=built").maskKinds).toEqual([...BUILT_MASK_KINDS]);
    expect(optionsFromLocation("?mockMasks=radial,linear").maskKinds).toEqual(["linear", "radial"]);
    expect(parseMockMasks("lasso")).toBeNull();
    expect(parseMockMasks("linear,lasso")).toBeNull();
    expect(parseMockMasks(null)).toBeNull();
  });

  it("answers from the options it has, and a fresh answer each time", async () => {
    const a = createMockApi({ speed: Number.POSITIVE_INFINITY, maskKinds: ["radial"] });
    api = a;
    const first = await a.maskKinds();
    first.push("brush");
    expect(await a.maskKinds()).toEqual(["radial"]);
  });
});

describe("viewport_map_points", () => {
  it("has no place for anything while no photo is shown", async () => {
    const a = createMockApi({ speed: Number.POSITIVE_INFINITY });
    api = a;
    expect(await a.viewportMapPoints([[0.5, 0.5]], "photo")).toEqual([null]);
    expect(await a.viewportMapPoints([[10, 10]], "cell")).toEqual([null]);
  });

  it("maps the photo frame onto the picture in the cell, and back", async () => {
    const a = await open();
    // The photo is 3:2 and so is the cell: its corners are the cell's.
    expect(await a.viewportMapPoints([[0, 0], [1, 1], [0.5, 0.25]], "photo")).toEqual([[0, 0], [600, 400], [300, 100]]);
    expect(await a.viewportMapPoints([[0, 0], [600, 400], [300, 100]], "cell")).toEqual([[0, 0], [1, 1], [0.5, 0.25]]);
    // Outside the photo a point is still drawn, outside the cell.
    expect(await a.viewportMapPoints([[-0.5, 1.5]], "photo")).toEqual([[-300, 600]]);
  });

  it("has no place in the photo for the paper around a letterboxed picture, and none for a number that is not", async () => {
    const a = await open({ nodes: { input: { orientation: 90 } } });
    // Turned a quarter, the picture is 2:3 in a 3:2 cell: paper to the left and right.
    expect(await a.viewportMapPoints([[10, 200], [300, 200], [Number.NaN, 5]], "cell")).toEqual([null, [0.5, 0.5], null]);
    expect(await a.viewportMapPoints([[0.5, Number.POSITIVE_INFINITY]], "photo")).toEqual([null]);
  });

  it("follows the turn and the crop, and each direction is the other's inverse", async () => {
    const crop = { rect: { x: 0.1, y: 0.2, width: 0.6, height: 0.5 }, angle_deg: 0, aspect: "free" as const };
    for (const settings of [{ nodes: { input: { orientation: 90 as const } } }, { nodes: { input: { orientation: 270 as const, crop } } }, { nodes: { input: { crop } } }]) {
      const a = await open(settings);
      const points: Array<[number, number]> = [[0.3, 0.4], [0.5, 0.5], [0.7, 0.25]];
      const css = (await a.viewportMapPoints(points, "photo")) as Array<[number, number]>;
      const back = await a.viewportMapPoints(css, "cell");
      back.forEach((p, i) => {
        if (p === null) return;
        expect(p[0]).toBeCloseTo(points[i]![0], 9);
        expect(p[1]).toBeCloseTo(points[i]![1], 9);
      });
      expect(back.filter((p) => p !== null).length).toBeGreaterThan(0);
      a.reset();
    }
  });

  it("puts a quarter turn clockwise where the camera's top left corner goes: the cell's top right", async () => {
    const a = await open({ nodes: { input: { orientation: 90 } } });
    // The turned picture is 2:3, 266.7 × 400 in the middle of the 600 × 400 cell (166.7 to 433.3 across).
    const [corner] = await a.viewportMapPoints([[0, 0]], "photo");
    expect(corner?.[0]).toBeCloseTo(600 * (0.5 + 2 / 9), 9);
    expect(corner?.[1]).toBeCloseTo(0, 9);
  });
});

describe("Copy settings carries the rows (docs/23 §3.8)", () => {
  const rows: MaskItem[] = [
    { ...newMask("object"), points: [[0.2, 0.3, true]], box: [0.1, 0.1, 0.5, 0.5] } as MaskItem,
    newMask("subject"),
    { ...newMask("radial"), center: [0.2, 0.7] } as MaskItem,
    newMask("color_range"),
  ];

  it("gives an Object row no points and no box, and every other row as it is", async () => {
    const a = await open({
      nodes: { "2": { kind: "corrector", name: "Mixed", pos: [1, 0], masks: { invert: true, items: rows } } },
      links: [
        { from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } },
        { from: { node: 1, port: "picture" }, to: { node: 2, port: "picture" } },
        { from: { node: 2, port: "picture" }, to: { node: "output", port: "picture" } },
      ],
    });
    const id = (await a.sessionState()).photos[3]!.id;
    const copied = await a.copySettings(id);
    const node = copied.nodes?.["2"]?.masks;
    expect(node?.invert).toBe(true);
    expect(node?.items?.[0]).toEqual({ ...rows[0], points: [], box: null });
    expect(node?.items?.slice(1)).toEqual(rows.slice(1));
    // The mock's own rule and the one the page's tests use are the same function.
    expect(copiedPatch(await a.getSettings(id))).toEqual(copied);
  });
});

describe("the blur map of a Background row (docs/23 §7.6: \"Finding…\")", () => {
  const BACKGROUND_ON_4: SettingsPatch = { nodes: { "4": { kind: "corrector", name: "Sky", pos: [1, 0], masks: { invert: false, items: [newMask("background")] } } }, links: [] };
  const busy = (a: MockApi): Array<{ busy: boolean; cpu: boolean; finding: boolean }> => {
    const seen: Array<{ busy: boolean; cpu: boolean; finding: boolean }> = [];
    a.on("viewport_busy", (e) => seen.push(e));
    return seen;
  };
  const tick = (ms = 10): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

  it("is made on the first frame of a photo that has a Background row with the eye on, and said once: finding true while busy, then false", async () => {
    const a = await open();
    const seen = busy(a);
    // No row, no map: a frame says nothing.
    await a.viewportSetRect({ x: 0, y: 0, width: 640, height: 400, scale: 1 });
    await tick();
    expect(seen).toEqual([]);
    // The row is added: the next frame makes the map.
    const id = (await a.sessionState()).photos[3]!.id;
    await a.updateSettings(id, { nodes: { "1": { masks: { items: [newMask("background")] } } } });
    await tick(20);
    expect(seen).toEqual([
      { busy: true, cpu: false, finding: true },
      { busy: false, cpu: false, finding: false },
    ]);
    // The photo has its map now: another change of the row says nothing more.
    await a.updateSettings(id, { nodes: { "1": { masks: { edit: { "0": { level: 0.8 } } } } } });
    await tick(20);
    expect(seen).toHaveLength(2);
  });

  it("says nothing for a row with the eye off, and with the rows already there it is the photo's first frame that finds", async () => {
    const off: SettingsPatch = { nodes: { "4": { kind: "corrector", name: "Sky", pos: [1, 0], masks: { invert: false, items: [{ ...newMask("background"), enabled: false }] } } }, links: [] };
    const a = await open(off);
    const seen = busy(a);
    await a.viewportSetRect({ x: 0, y: 0, width: 600, height: 401, scale: 1 });
    await tick();
    expect(seen).toEqual([]);
    a.reset();
    // With the row already there, the photo's first frame is the one that finds the map.
    const b = createMockApi({ speed: Number.POSITIVE_INFINITY, initialSettings: { DSC04204: BACKGROUND_ON_4 } });
    api = b;
    const found = busy(b);
    await b.openPaths(["/photos"]);
    b.flush();
    await b.setCurrent((await b.sessionState()).photos[3]!.id);
    await b.viewportSetRect({ x: 0, y: 0, width: 600, height: 402, scale: 1 });
    await tick(20);
    expect(found.filter((e) => e.busy)).toEqual([{ busy: true, cpu: false, finding: true }]);
    expect(found.at(-1)).toEqual({ busy: false, cpu: false, finding: false });
  });
});

describe("Subject and People (docs/23 §4.5 and §7.6, N-19)", () => {
  const ROWS: SettingsPatch = { nodes: { "1": { masks: { items: [newMask("subject"), newMask("people")] } } } };
  const events = (a: MockApi): { busy: Array<{ busy: boolean; cpu: boolean; finding: boolean }>; states: Array<{ photo_id: number; nothing_found: string[] }> } => {
    const seen = { busy: [] as Array<{ busy: boolean; cpu: boolean; finding: boolean }>, states: [] as Array<{ photo_id: number; nothing_found: string[] }> };
    a.on("viewport_busy", (e) => seen.busy.push(e));
    a.on("vision_state", (e) => seen.states.push(e));
    return seen;
  };
  const tick = (ms = 20): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));
  /** The newest picture the mock's fallback viewport sent, decoded. */
  const pictures = (a: MockApi): (() => string) => {
    let url = "";
    a.on("viewport_frame", (f) => (url = f.url));
    return () => decodeURIComponent(url.replace(/^data:image\/svg\+xml;charset=utf-8,/, ""));
  };

  it("makes the masks in the background when the rows appear: finding while it works, then vision_state and not again", async () => {
    const a = await open();
    const seen = events(a);
    const id = (await a.sessionState()).photos[3]!.id;
    await a.updateSettings(id, ROWS);
    await tick();
    // DSC04204 has a figure on its near ground, so both requests find something.
    expect(seen.busy).toEqual([
      { busy: true, cpu: false, finding: true },
      { busy: false, cpu: false, finding: false },
    ]);
    expect(seen.states).toEqual([{ photo_id: id, nothing_found: [] }]);
    // The photo has its masks: another change of the rows says nothing more.
    await a.updateSettings(id, { nodes: { "1": { masks: { edit: { "0": { feather: 0.3 } } } } } });
    await tick();
    expect(seen.busy).toHaveLength(2);
    expect(seen.states).toHaveLength(1);
  });

  it("finds nothing for People in a scene with no person, and says so", async () => {
    const a = createMockApi({ speed: Number.POSITIVE_INFINITY, initialSettings: { DSC04201: ROWS } });
    api = a;
    const seen = events(a);
    await a.openPaths(["/photos"]);
    a.flush();
    const first = (await a.sessionState()).photos[0]!;
    await a.setCurrent(first.id);
    await a.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    await tick();
    expect(seen.states).toEqual([{ photo_id: first.id, nothing_found: ["people"] }]);
  });

  it("makes nothing on a computer that cannot (mask_kinds leaves the kinds out)", async () => {
    const a = createMockApi({ speed: Number.POSITIVE_INFINITY, maskKinds: ["object", "background", "linear"], initialSettings: { DSC04204: ROWS } });
    api = a;
    const seen = events(a);
    await a.openPaths(["/photos"]);
    a.flush();
    await a.setCurrent((await a.sessionState()).photos[3]!.id);
    await a.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    await tick();
    expect(seen.busy.filter((e) => e.finding)).toEqual([]);
    expect(seen.states).toEqual([]);
  });

  it("draws the far hill for a Subject and a figure for People in the Mask only picture once the masks are made, and nothing before", async () => {
    const a = createMockApi({ speed: 1, viewport: "fallback", initialSettings: { DSC04204: ROWS } });
    api = a;
    const picture = pictures(a);
    await a.openPaths(["/photos"]);
    a.flush();
    await a.setCurrent((await a.sessionState()).photos[3]!.id);
    await a.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
    await a.viewportSetMode({ kind: "fit" }, { node: 1, style: "mask_only" });
    await new Promise((resolve) => setTimeout(resolve, 50));
    // A viewer never waits: the rows select nothing until the masks are there.
    expect(picture()).toContain('data-mask-view="mask_only"');
    expect(picture()).not.toContain("data-vision-row");
    await new Promise((resolve) => setTimeout(resolve, 700));
    const drawn = picture();
    expect(drawn).toContain('data-vision-row="subject"');
    expect(drawn).toContain('data-vision-row="people"');
  });
});
