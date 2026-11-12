/**
 * The Face Refinement node's interface (`docs/27` S-11c, `design/skin/`): the Add node item, the card, the head's line in each phase, the Face tab's cards and
 * the patches they send (`all` against `faces`, Same for all faces, Copy to other faces, each card's Reset), Protect's strokes and Clear, the choice of a face
 * from the list and from the photo, and the states. The screen runs on the mock, as in a browser.
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import type { NodeLink, SettingsPatch } from "../../api/types";
import { defaultFaceSettings, nodeOf } from "../../state/settingsGraph";
import type { FaceRefinementItem } from "../../state/settingsGraph";
import { openPhotos, openTab, renderApp, tools, valueOf } from "../../test/harness";
import type { Harness, UpdateSpy } from "../../test/harness";
import { mockRect } from "../../test/raf";

let h: Harness | undefined;
let windowHeight = window.innerHeight;
beforeEach(() => {
  windowHeight = window.innerHeight;
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true, writable: true });
});
afterEach(() => {
  h?.unmount();
  h = undefined;
  Object.defineProperty(window, "innerHeight", { value: windowHeight, configurable: true, writable: true });
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });

/** Main, then a Face Refinement node (2), then the Output. */
const WITH_FACES = (more: Record<string, unknown> = {}): SettingsPatch => ({
  nodes: { "2": { kind: "face_refinement", name: "Face Refinement", pos: [1, 0], ...more } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
});

/** DSC04204 in the node view with the Face Refinement node selected, the photo cell 600 × 400 and the mock told so. */
async function setup(settings: SettingsPatch = WITH_FACES(), options: MockOptions = {}, tab: string | null = null, beforeOpen?: (api: Harness["api"]) => void): Promise<UpdateSpy> {
  h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: settings }, ...options });
  beforeOpen?.(h.api);
  await openPhotos(h);
  await openTab(h, "Light");
  await h.user.keyboard("n");
  await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
  fireEvent.click(screen.getByRole("button", { name: /^Node 2, Face Refinement/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Face Refinement/ })).toHaveAttribute("aria-pressed", "true"));
  const cell = document.querySelector<HTMLElement>("[data-viewport]")!;
  mockRect(cell, { left: 0, top: 0, width: 600, height: 400 });
  await h.api.viewportSetRect({ x: 0, y: 0, width: 600, height: 400, scale: 1 });
  await act(async () => {
    fireEvent(window, new Event("resize"));
  });
  if (tab) await h.user.click(within(tools()).getByRole("tab", { name: tab }));
  return vi.spyOn(h.api, "updateSettings");
}

const patches = (spy: UpdateSpy): SettingsPatch[] => spy.mock.calls.map((c) => c[1] as SettingsPatch);
const node = (): FaceRefinementItem => nodeOf(h!.store.currentSettings()!.settings, 2) as FaceRefinementItem;
const head = (): HTMLElement => within(tools()).getByRole("group", { name: "Node" });
const line = (): string => head().querySelector(".ym-nodehead__line")?.textContent ?? "";
const card = (name: string): HTMLElement => within(tools()).getByRole("region", { name });
const cardTitles = (): string[] => within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent ?? "");
const faceButtons = (): HTMLElement[] => within(card("Faces")).getAllByRole("button", { name: /^Face \d+$/ });
async function type(label: string, text: string): Promise<void> {
  const box = screen.getByRole("textbox", { name: `${label} value` });
  await h!.user.click(box);
  await h!.user.clear(box);
  await h!.user.type(box, `${text}{Enter}`);
}
/** The tab is ready when the Faces card says how many faces there are. */
const ready = (n: number): Promise<HTMLElement> => waitFor(() => expect(within(tools()).getByText(`${n} found`)).toBeInTheDocument()).then(() => card("Faces"));

describe("Add node › Face Refinement", () => {
  it("is an item with its line after Film, before the rule, and adds a numbered node after the selected one", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4] });
    await openPhotos(h);
    await openTab(h, "Light");
    await h.user.keyboard("n");
    await waitFor(() => expect(screen.getByRole("button", { name: "Nodes" })).toHaveAttribute("aria-pressed", "true"));
    await h.user.click(screen.getByRole("button", { name: "Add node" }));
    const menu = await screen.findByRole("menu", { name: "Add node" });
    const items = within(menu).getAllByRole("menuitem").map((i) => i.textContent);
    expect(items.slice(0, 3)).toEqual(["CorrectorGrades all or part of the photo.", "FilmModels a colour negative, a print and a scan.", "Face RefinementRefines each face: skin, under-eyes and teeth."]);
    expect(menu.querySelectorAll('[role="separator"]').length).toBe(1);
    const separator = menu.querySelector('[role="separator"]')!;
    expect(items.indexOf("Face RefinementRefines each face: skin, under-eyes and teeth.")).toBeLessThan(
      within(menu).getAllByRole("menuitem").findIndex((i) => separator.compareDocumentPosition(i) & Node.DOCUMENT_POSITION_FOLLOWING),
    );
    await h.user.click(within(menu).getByRole("menuitem", { name: /^Face Refinement/ }));
    await waitFor(() => expect(screen.getByRole("button", { name: /^Node 2, Face Refinement/ })).toHaveAttribute("aria-pressed", "true"));
    expect(node()).toMatchObject({ kind: "face_refinement", same_for_all: true, faces: [], all: defaultFaceSettings() });
    // The panel opens on its first tab.
    expect(within(tools()).getAllByRole("tab").map((t) => t.textContent)).toEqual(["Face", "Mask"]);
    expect(within(tools()).getByRole("tab", { name: "Face" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("the card", () => {
  it("is numbered like Film, called Face Refi…, has a face glyph in its footer and the node's ports", async () => {
    await setup();
    const box = screen.getByRole("button", { name: /^Node 2, Face Refinement/ });
    expect(box.querySelector(".ym-gnode__num")).toHaveTextContent("02");
    expect(box.closest(".ym-gnode")!.querySelector("[data-node-name]")).toHaveTextContent("Face Refi…");
    expect(box.querySelector('[data-glyph="face"] svg')).not.toBeNull();
    expect(box.closest(".ym-gnode")!.querySelectorAll("[data-port]").length).toBeGreaterThanOrEqual(4);
  });

  it("has the bypassed state", async () => {
    await setup(WITH_FACES({ bypass: true }));
    const gnode = screen.getByRole("button", { name: /^Node 2, Face Refinement/ }).closest(".ym-gnode")!;
    expect(gnode).toHaveAttribute("data-bypass");
    expect(gnode.querySelector('[data-glyph="bypass"]')).not.toBeNull();
  });

  it("has the dead state when it is not on a path to the Output", async () => {
    await setup({
      nodes: { "2": { kind: "face_refinement", name: "Face Refinement", pos: [1, 1] } },
      links: [picture("input", 1), picture(1, "output")],
    });
    expect(screen.getByRole("button", { name: /^Node 2, Face Refinement, not connected/ }).closest(".ym-gnode")).toHaveAttribute("data-dead");
  });
});

describe("the head's line", () => {
  it("says how many faces there are", async () => {
    await setup(WITH_FACES(), { faces: "five" });
    await ready(5);
    expect(within(head()).getByRole("heading", { level: 2, name: "Face Refinement" })).toBeInTheDocument();
    expect(head().querySelector(".ym-nodehead__chip")).toHaveTextContent("02");
    expect(line()).toBe("5 faces");
  });

  it("says 1 face", async () => {
    await setup(WITH_FACES(), { faces: "one" });
    await waitFor(() => expect(line()).toBe("1 face"));
  });

  it("says No faces found", async () => {
    await setup(WITH_FACES(), { faces: "none" });
    await waitFor(() => expect(line()).toBe("No faces found"));
  });

  it("says Finding faces while the worker runs", async () => {
    await setup(WITH_FACES(), { speed: 0.0001 });
    expect(line()).toBe("Finding faces");
  });

  it("says plainly when the worker failed or the models are missing", async () => {
    await setup(WITH_FACES(), { faces: "fail" });
    await waitFor(() => expect(line()).toBe("Can't find faces"));
    h!.unmount();
    h = undefined;
    await setup(WITH_FACES(), { faces: "missing" });
    await waitFor(() => expect(line()).toBe("Face models missing"));
  });
});

describe("the states", () => {
  it("shows Finding faces… with its line and a bar, and nothing to set", async () => {
    await setup(WITH_FACES(), { speed: 0.0001 });
    const faces = card("Faces");
    expect(within(faces).getByText("Finding faces…")).toBeInTheDocument();
    expect(within(faces).getByText("The first time takes a few seconds. The photo stays as it is until then.")).toBeInTheDocument();
    expect(within(faces).getByRole("progressbar", { name: "Finding faces" })).toBeInTheDocument();
    expect(cardTitles()).toEqual(["Faces"]);
    expect(within(tools()).queryByRole("slider")).toBeNull();
  });

  it("shows No faces found in this photo. with what to do, and nothing to set", async () => {
    await setup(WITH_FACES(), { faces: "none" });
    await within(tools()).findByText("No faces found in this photo.");
    expect(
      within(tools()).getByText(
        "Face Refinement works on faces that are at least 200 px across and turned towards the camera. Try another photo, or delete this node. It changes nothing until it finds a face.",
      ),
    ).toBeInTheDocument();
    expect(cardTitles()).toEqual(["Faces"]);
  });

  it("shows the failure as the core worded it", async () => {
    await setup(WITH_FACES(), { faces: "fail" });
    expect(await within(tools()).findByText("Faces can't be found: the model stopped before it finished.")).toBeInTheDocument();
    expect(cardTitles()).toEqual(["Faces"]);
  });

  it("shows the missing models as the core worded it", async () => {
    await setup(WITH_FACES(), { faces: "missing" });
    expect(await within(tools()).findByText("Face Refinement needs the face models, and this copy of Yellowmoon has none. Reinstall the app.")).toBeInTheDocument();
    expect(cardTitles()).toEqual(["Faces"]);
  });
});

describe("the Faces card", () => {
  it("lists the faces as numbered items, five to a row, with Same for all faces on and the first face chosen", async () => {
    await setup(WITH_FACES(), { faces: "five" });
    const faces = await ready(5);
    expect(cardTitles()).toEqual(["Faces", "All faces", "Skin grading", "Teeth", "Protect"]);
    expect(faceButtons().map((b) => b.textContent)).toEqual(["1", "2", "3", "4", "5"]);
    expect(within(faces).getByRole("switch", { name: "Same for all faces" })).toBeChecked();
    expect(faces.querySelector(".ym-faces")).not.toHaveAttribute("data-scroll");
  });

  it("scrolls the list from eleven faces", async () => {
    await setup(WITH_FACES(), { faces: "twelve" });
    const faces = await ready(12);
    expect(faceButtons().length).toBe(12);
    expect(faces.querySelector(".ym-faces")).toHaveAttribute("data-scroll");
  });

  it("has no list and no switch with one face, and the card says Face", async () => {
    await setup(WITH_FACES(), { faces: "one" });
    await ready(1);
    expect(within(card("Faces")).queryByRole("switch")).toBeNull();
    expect(cardTitles()).toEqual(["Faces", "Face", "Skin grading", "Teeth", "Protect"]);
    expect(within(card("Skin grading")).getByText("Face 1")).toBeInTheDocument();
    expect(within(card("Skin grading")).queryByText("All faces")).toBeNull();
  });

  it("chooses a face from the list: the card and the tags follow, nothing is sent", async () => {
    const spy = await setup(WITH_FACES({ same_for_all: false }), { faces: "five" });
    await ready(5);
    await h!.user.click(faceButtons()[2]!);
    expect(faceButtons()[2]).toHaveAttribute("aria-pressed", "true");
    expect(faceButtons()[0]).toHaveAttribute("aria-pressed", "false");
    expect(cardTitles()).toEqual(["Faces", "Face 3", "Skin grading", "Teeth", "Protect"]);
    expect(within(card("Teeth")).getByText("Face 3")).toBeInTheDocument();
    expect(within(card("Teeth")).getByText("Copies all of Face 3's settings.")).toBeInTheDocument();
    expect(patches(spy)).toEqual([]);
  });
});

describe("the faces' pictures", () => {
  it("shows the picture the core answers, with the number on a badge", async () => {
    await setup(WITH_FACES(), { faces: "ok" });
    await ready(2);
    await waitFor(() => expect(faceButtons()[0]!.querySelector("img")).not.toBeNull());
    expect(faceButtons()[0]!.querySelector("img")!.getAttribute("src")).toMatch(/^data:image\/svg/);
    expect(faceButtons()[0]!.querySelector(".ym-face__num")!.textContent).toBe("1");
  });

  it("keeps the number on the neutral disc when the picture cannot be made, and the panel stays usable", async () => {
    await setup(WITH_FACES(), { faces: "five" }, null, (api) => {
      vi.spyOn(api, "faceCrop").mockRejectedValue(new Error("no"));
    });
    await ready(5);
    await waitFor(() => expect(h!.api.faceCrop).toHaveBeenCalled());
    expect(faceButtons().every((b) => b.querySelector("img") === null)).toBe(true);
    expect(faceButtons()[1]!.querySelector(".ym-face__disc")!.textContent).toBe("2");
    await h!.user.click(faceButtons()[1]!);
    expect(faceButtons()[1]).toHaveAttribute("aria-pressed", "true");
  });

  it("asks once for a face while face_state is the same", async () => {
    await setup(WITH_FACES(), { faces: "five" }, null, (api) => {
      vi.spyOn(api, "faceCrop");
    });
    await ready(5);
    await waitFor(() => expect(h!.api.faceCrop).toHaveBeenCalledTimes(5));
    await h!.user.click(faceButtons()[0]!);
    expect(h!.api.faceCrop).toHaveBeenCalledTimes(5);
  });
});

describe("Show skin areas", () => {
  const eye = (): HTMLElement => within(tools()).getByRole("button", { name: "Show skin areas" });
  const views = (): unknown[] => vi.mocked(h!.api.viewportSetMode).mock.calls.map((c) => c[1] ?? null);

  async function withEye(): Promise<void> {
    await setup(WITH_FACES({ same_for_all: false }), { faces: "five" });
    await ready(5);
    vi.spyOn(h!.api, "viewportSetMode");
  }

  it("sends the overlay view of the chosen face, and follows another face", async () => {
    await withEye();
    await h!.user.click(eye());
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay", face: 1 }));
    await h!.user.click(faceButtons()[2]!);
    await waitFor(() => expect(views().at(-1)).toEqual({ node: 2, style: "overlay", face: 3 }));
    await h!.user.click(eye());
    await waitFor(() => expect(views().at(-1)).toBeNull());
    expect(h!.store.getState().skinAreas).toBeNull();
  });

  it("ends when another node is selected", async () => {
    await withEye();
    await h!.user.click(eye());
    await waitFor(() => expect(h!.store.getState().skinAreas).toBe(1));
    fireEvent.click(screen.getByRole("button", { name: /^Node 1/ }));
    await waitFor(() => expect(h!.store.getState().skinAreas).toBeNull());
    expect(views().at(-1)).toBeNull();
  });

  it("is exclusive with Show mask", async () => {
    await withEye();
    await h!.user.click(eye());
    await waitFor(() => expect(h!.store.getState().skinAreas).toBe(1));
    h!.store.setMaskView({ on: true });
    expect(h!.store.getState().skinAreas).toBeNull();
    h!.store.setSkinAreas(2);
    expect(h!.store.getState().maskView.on).toBe(false);
    expect(h!.store.getState().skinAreas).toBe(2);
  });

  it("shows the legend only while the eye is on", async () => {
    await withEye();
    expect(screen.queryByRole("list", { name: "Skin areas" })).toBeNull();
    await h!.user.click(eye());
    const legend = await screen.findByRole("list", { name: "Skin areas" });
    expect(within(legend).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["Skin", "Under the eyes", "Even skin and grading", "Teeth"]);
    await h!.user.click(eye());
    expect(screen.queryByRole("list", { name: "Skin areas" })).toBeNull();
  });
});

describe("the faces on the photo", () => {
  const ovals = (): HTMLElement[] => [...document.querySelectorAll<HTMLElement>("[data-face-number]")];

  it("draws an oval for each face, the chosen one yellow, and a click on a face chooses it", async () => {
    await setup(WITH_FACES({ same_for_all: false }), { faces: "five" });
    await ready(5);
    await waitFor(() => expect(ovals().length).toBe(5));
    expect(ovals().map((o) => o.dataset.faceNumber)).toEqual(["1", "2", "3", "4", "5"]);
    expect(ovals().filter((o) => o.hasAttribute("data-selected")).map((o) => o.dataset.faceNumber)).toEqual(["1"]);
    fireEvent.click(document.querySelector('[data-face-number="4"] [data-face-hit]')!);
    await waitFor(() => expect(faceButtons()[3]).toHaveAttribute("aria-pressed", "true"));
    expect(ovals().filter((o) => o.hasAttribute("data-selected")).map((o) => o.dataset.faceNumber)).toEqual(["4"]);
    expect(within(tools()).getByRole("heading", { level: 3, name: "Face 4" })).toBeInTheDocument();
  });

  it("draws every oval as chosen while Same for all faces is on, and none on the Mask tab", async () => {
    await setup(WITH_FACES(), { faces: "five" });
    await ready(5);
    await waitFor(() => expect(ovals().length).toBe(5));
    expect(ovals().every((o) => o.hasAttribute("data-selected"))).toBe(true);
    await h!.user.click(within(tools()).getByRole("tab", { name: "Mask" }));
    await waitFor(() => expect(ovals().length).toBe(0));
  });
});

describe("the Face card and its patches", () => {
  it("writes `all` while Same for all faces is on, and each changed value gets its own Reset", async () => {
    const spy = await setup(WITH_FACES(), { faces: "five" });
    await ready(5);
    expect(valueOf("Blemish")).toBe("30");
    await type("Blemish", "45");
    await waitFor(() => expect(node().all.blemish).toBe(45));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { all: { blemish: 45 } } } });
    expect(node().faces).toEqual([]);
    const row = screen.getByRole("slider", { name: "Blemish" }).closest("[data-slider]")!;
    expect(row).toHaveAttribute("data-changed");
    await h!.user.click(within(row as HTMLElement).getByRole("button", { name: "Reset Blemish" }));
    await waitFor(() => expect(node().all.blemish).toBe(30));
  });

  it("has Contrast and Texture bipolar and the others from 0 to 100", async () => {
    await setup(WITH_FACES(), { faces: "one" });
    await ready(1);
    expect(screen.getByRole("slider", { name: "Texture" })).toHaveAttribute("aria-valuemin", "-100");
    expect(screen.getByRole("slider", { name: "Contrast" })).toHaveAttribute("aria-valuemin", "-100");
    for (const name of ["Blemish", "Dark circles", "Even skin", "Shadow desaturation", "Shine removal", "Whitening"]) {
      expect(screen.getByRole("slider", { name })).toHaveAttribute("aria-valuemin", "0");
      expect(screen.getByRole("slider", { name })).toHaveAttribute("aria-valuemax", "100");
    }
  });

  it("turning Same for all faces off starts every face from `all`, and a face then has its own value", async () => {
    const spy = await setup(WITH_FACES({ all: { blemish: 50 } }), { faces: "five" });
    await ready(5);
    await h!.user.click(within(card("Faces")).getByRole("switch", { name: "Same for all faces" }));
    await waitFor(() => expect(node().same_for_all).toBe(false));
    expect(node().faces).toHaveLength(5);
    expect(node().faces.every((f) => f.blemish === 50)).toBe(true);
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { same_for_all: false, faces: node().faces } } });
    expect(cardTitles()).toContain("Face 1");
    await h!.user.click(faceButtons()[1]!);
    await type("Blemish", "70");
    await waitFor(() => expect(node().faces[1]!.blemish).toBe(70));
    expect(node().faces.map((f) => f.blemish)).toEqual([50, 70, 50, 50, 50]);
    expect(node().all.blemish).toBe(50);
  });

  it("keeps a list it already has when Same for all faces is turned off, and `all` when it is turned on again", async () => {
    const mine = { ...defaultFaceSettings(), blemish: 5 };
    await setup(WITH_FACES({ same_for_all: false, faces: [mine, mine] }), { faces: "five" });
    await ready(5);
    expect(valueOf("Blemish")).toBe("5");
    await h!.user.click(within(card("Faces")).getByRole("switch", { name: "Same for all faces" }));
    await waitFor(() => expect(node().same_for_all).toBe(true));
    expect(node().faces).toEqual([mine, mine]);
    expect(valueOf("Blemish")).toBe("30");
  });

  it("the Face card's Reset puts its four rows back to the defaults and leaves grading and teeth alone", async () => {
    await setup(WITH_FACES({ all: { blemish: 80, texture: -20, skin_contrast: 10, teeth_whitening: 40 } }), { faces: "one" });
    await ready(1);
    await h!.user.click(within(card("Face")).getAllByRole("button", { name: "Reset" }).at(-1)!);
    await waitFor(() => expect(node().all.blemish).toBe(30));
    expect(node().all).toEqual({ ...defaultFaceSettings(), skin_contrast: 10, teeth_whitening: 40 });
  });

  it("Skin grading's Reset sets its three to 0, and Teeth's sets Whitening to 0", async () => {
    await setup(WITH_FACES({ all: { blemish: 80, skin_contrast: 10, shadow_desaturation: 20, shine_removal: 30, teeth_whitening: 40 } }), { faces: "one" });
    await ready(1);
    await h!.user.click(within(card("Skin grading")).getAllByRole("button", { name: "Reset" }).at(-1)!);
    await waitFor(() => expect(node().all.skin_contrast).toBe(0));
    expect(node().all).toMatchObject({ blemish: 80, skin_contrast: 0, shadow_desaturation: 0, shine_removal: 0, teeth_whitening: 40 });
    await h!.user.click(within(card("Teeth")).getAllByRole("button", { name: "Reset" }).at(-1)!);
    await waitFor(() => expect(node().all.teeth_whitening).toBe(0));
    expect(node().all.blemish).toBe(80);
  });

  it("Copy to other faces gives every other face this face's settings, only with several faces and Same for all faces off", async () => {
    await setup(WITH_FACES(), { faces: "five" });
    await ready(5);
    expect(within(tools()).queryByRole("button", { name: "Copy to other faces" })).toBeNull();
    await h!.user.click(within(card("Faces")).getByRole("switch", { name: "Same for all faces" }));
    await waitFor(() => expect(node().same_for_all).toBe(false));
    await h!.user.click(faceButtons()[1]!);
    await type("Blemish", "66");
    await type("Whitening", "20");
    await waitFor(() => expect(node().faces[1]!.teeth_whitening).toBe(20));
    expect(node().faces[0]!.blemish).toBe(30);
    await h!.user.click(within(tools()).getByRole("button", { name: "Copy to other faces" }));
    await waitFor(() => expect(node().faces[0]!.blemish).toBe(66));
    expect(node().faces).toHaveLength(5);
    for (const f of node().faces) expect(f).toMatchObject({ blemish: 66, teeth_whitening: 20 });
    expect(node().all.blemish).toBe(30);
  });

  it("the head's Reset returns Same for all faces, `all` and `faces` to the defaults and keeps protection", async () => {
    const strokes = [{ size: 0.03, hardness: 0.5, erase: false, points: [[0.3, 0.3], [0.31, 0.32]] }];
    await setup(WITH_FACES({ same_for_all: false, all: { blemish: 80 }, faces: [{ ...defaultFaceSettings(), blemish: 5 }], protect: { strokes } }), { faces: "five" });
    await ready(5);
    await h!.user.click(within(head()).getByRole("button", { name: "Reset this node" }));
    await waitFor(() => expect(node().same_for_all).toBe(true));
    expect(node().faces).toEqual([]);
    expect(node().all).toEqual(defaultFaceSettings());
    expect(node().protect.strokes).toEqual(strokes);
  });
});

describe("Protect", () => {
  const layer = (): HTMLElement | null => document.querySelector<HTMLElement>("[data-brush-layer]");
  function drag(places: Array<[number, number]>): void {
    const [first, ...rest] = places as [[number, number], ...Array<[number, number]>];
    fireEvent.pointerDown(layer()!, { clientX: first[0], clientY: first[1], button: 0, buttons: 1, pointerId: 1 });
    for (const [x, y] of rest) fireEvent.pointerMove(window, { clientX: x, clientY: y, pointerId: 1 });
    const last = rest.at(-1) ?? first;
    fireEvent.pointerUp(window, { clientX: last[0], clientY: last[1], pointerId: 1 });
  }

  it("says what it is for, and Paint turns the tool on with the hint", async () => {
    await setup(WITH_FACES(), { faces: "one" });
    await ready(1);
    const protect = card("Protect");
    expect(within(protect).getByText("Paint over moles, freckles and scars you want to keep.")).toBeInTheDocument();
    expect(layer()).toBeNull();
    await h!.user.click(within(protect).getByRole("radio", { name: "Paint" }));
    await waitFor(() => expect(h!.store.getState().brushTool).toBe(true));
    expect(within(protect).getByText("Paint on the photo. Hold ⌥ to erase.")).toBeInTheDocument();
    await waitFor(() => expect(layer()).not.toBeNull());
    // The tool and the faces are not on the photo together.
    expect(document.querySelector("[data-face-layer]")).toBeNull();
  });

  it("sends a stroke on release as `protect.strokes_append`", async () => {
    const spy = await setup(WITH_FACES(), { faces: "one" });
    await ready(1);
    await h!.user.click(within(card("Protect")).getByRole("radio", { name: "Paint" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    drag([[100, 100], [140, 120], [180, 140]]);
    await waitFor(() => expect(node().protect.strokes).toHaveLength(1));
    const sent = patches(spy).at(-1)!;
    expect(Object.keys(sent.nodes!["2"]!)).toEqual(["protect"]);
    expect(Object.keys(sent.nodes!["2"]!.protect!)).toEqual(["strokes_append"]);
    expect(node().protect.strokes[0]).toMatchObject({ size: 0.03, hardness: 0.5, erase: false });
  });

  it("⌥ turns a stroke into an erasing one", async () => {
    await setup(WITH_FACES(), { faces: "one" });
    await ready(1);
    await h!.user.click(within(card("Protect")).getByRole("radio", { name: "Paint" }));
    await waitFor(() => expect(layer()).not.toBeNull());
    fireEvent.pointerDown(layer()!, { clientX: 100, clientY: 100, button: 0, buttons: 1, pointerId: 1, altKey: true });
    fireEvent.pointerMove(window, { clientX: 140, clientY: 120, pointerId: 1 });
    fireEvent.pointerUp(window, { clientX: 140, clientY: 120, pointerId: 1 });
    await waitFor(() => expect(node().protect.strokes).toHaveLength(1));
    expect(node().protect.strokes[0]!.erase).toBe(true);
  });

  it("Clear empties the strokes, and is off with none", async () => {
    const strokes = [{ size: 0.03, hardness: 0.5, erase: false, points: [[0.3, 0.3], [0.31, 0.32]] as Array<[number, number]> }];
    const spy = await setup(WITH_FACES({ protect: { strokes } }), { faces: "one" });
    await ready(1);
    const clear = within(card("Protect")).getByRole("button", { name: "Clear" });
    await h!.user.click(clear);
    await waitFor(() => expect(node().protect.strokes).toEqual([]));
    expect(patches(spy).at(-1)).toEqual({ nodes: { "2": { protect: { strokes: [] } } } });
    expect(within(card("Protect")).getByRole("button", { name: "Clear" })).toBeDisabled();
  });
});

describe("the simple view", () => {
  it("has no Face Refinement controls for a photo with the node, and the existing note", async () => {
    h = renderApp({ current: 3, picked: [2, 3, 4], initialSettings: { DSC04204: WITH_FACES() } });
    await openPhotos(h);
    await openTab(h, "Light");
    expect(within(tools()).queryByRole("tab", { name: "Face" })).toBeNull();
    expect(within(tools()).queryByText("Faces")).toBeNull();
    expect(within(tools()).getByText(/These controls change/)).toBeInTheDocument();
  });
});
