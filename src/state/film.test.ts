import { describe, expect, it } from "vitest";
import type { DevelopSettings } from "../api/types";
import { cardStates, headLine, insertAfter, resetNode, rename, toggleBypass } from "./graphOps";
import { diffGraph, applyPatch, cardsOf, defaultSettings, filmsOf, foldPatches, newFilm, patchOf } from "./settingsGraph";
import type { FilmItem } from "./settingsGraph";

/** Input → Main → Film (2) → Output. */
function withFilm(): DevelopSettings {
  const base = defaultSettings();
  const film = newFilm(2, [1, 0]);
  return {
    nodes: {
      items: [...base.nodes.items.slice(0, 2), film, base.nodes.items[2]!],
      links: [
        { from: { node: "input", port: "picture" }, to: { node: 1, port: "picture" } },
        { from: { node: 1, port: "picture" }, to: { node: 2, port: "picture" } },
        { from: { node: 2, port: "picture" }, to: { node: "output", port: "picture" } },
      ],
    },
  };
}
const film = (s: DevelopSettings): FilmItem => filmsOf(s)[0]!;

describe("a Film node in the settings", () => {
  it("starts as Portra 400 on its matching paper, every number at its default", () => {
    expect(newFilm(2, [1, 0])).toEqual({
      kind: "film",
      id: 2,
      name: "Film",
      bypass: false,
      pos: [1, 0],
      film: { stock: "kodak_portra_400", exposure_ev: 0 },
      print: { paper: null, exposure_ev: 0, lights: { c: 0, m: 0, y: 0 } },
      halation: { amount: 100, size: 100 },
      grain: { amount: 100, size: 100, seed: 0 },
      strength: 100,
      masks: { invert: false, items: [] },
    });
  });

  it("takes a patch part by part, the lights one by one", () => {
    const s = applyPatch(withFilm(), { nodes: { "2": { print: { lights: { y: 5 } }, film: { stock: "kodak_ektar_100" } } } });
    const next = applyPatch(s, { nodes: { "2": { print: { lights: { m: -2 }, paper: "kodak_ektacolor_edge" } } } });
    expect(film(next).print).toEqual({ paper: "kodak_ektacolor_edge", exposure_ev: 0, lights: { c: 0, m: -2, y: 5 } });
    expect(film(next).film).toEqual({ stock: "kodak_ektar_100", exposure_ev: 0 });
    expect(film(next).grain.seed).toBe(0);
  });

  it("folds two patches of the lights in a row into both", () => {
    const folded = foldPatches(patchOf(2, { print: { lights: { y: 5 } } }), patchOf(2, { print: { lights: { m: 3 }, exposure_ev: 0.5 } }));
    expect(folded).toEqual({ nodes: { "2": { print: { lights: { y: 5, m: 3 }, exposure_ev: 0.5 } } } });
  });

  it("is added whole by a patch that names its kind, and undone by the diff", () => {
    const before = defaultSettings();
    const after = withFilm();
    const diff = diffGraph(before, after);
    expect(diff?.after.nodes?.["2"]).toMatchObject({ kind: "film", film: { stock: "kodak_portra_400" } });
    expect(applyPatch(before, diff!.after).nodes.items.find((i) => i.id === 2)).toEqual(film(after));
    expect(applyPatch(after, diff!.before).nodes.items.some((i) => i.id === 2)).toBe(false);
    const edited = applyPatch(after, patchOf(2, { halation: { amount: 200 } }));
    const back = diffGraph(after, edited)!;
    expect(back.after).toEqual({ nodes: { "2": { halation: { amount: 200, size: 100 } } } });
    expect(film(applyPatch(edited, back.before)).halation.amount).toBe(100);
  });
});

describe("a Film node in the graph", () => {
  it("has a card, a number and a place in the plan, like a corrector", () => {
    const s = withFilm();
    expect(cardsOf(s).map((c) => c.id)).toEqual([1, 2]);
    expect([...cardStates(s)].map(([id, state]) => [id, state.number, state.live])).toEqual([[1, 1, true], [2, 2, true]]);
  });

  it("is added after the selected node by Add node › Film, and takes the next number", () => {
    const out = insertAfter(defaultSettings(), 1, { kind: "film", name: "Film" });
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    expect(out.select).toBe(2);
    expect(out.patch.nodes?.["2"]).toMatchObject({ kind: "film", name: "Film", pos: [1, 0] });
    const s = applyPatch(defaultSettings(), out.patch);
    expect(s.nodes.links.map((l) => `${l.from.node}>${l.to.node}`)).toEqual(["input>1", "1>2", "2>output"]);
  });

  it("can be bypassed and renamed, and its quiet line says Film", () => {
    const s = withFilm();
    const bypass = toggleBypass(s, 2);
    expect(bypass).toMatchObject({ ok: true, patch: { nodes: { "2": { bypass: true } } } });
    expect(rename(s, 2, "  Portra  ")).toMatchObject({ ok: true, patch: { nodes: { "2": { name: "Portra" } } } });
    expect(headLine(s, 2)).toBe("Film · whole photo");
    expect(headLine(applyPatch(s, patchOf(2, { bypass: true })), 2)).toBe("Film · bypassed");
  });

  it("goes back to its defaults with Reset, and keeps its seed, name, masks and bypass", () => {
    const s = applyPatch(withFilm(), patchOf(2, { film: { stock: "fujifilm_velvia_100", exposure_ev: 1 }, strength: 40, grain: { amount: 10, seed: 7 }, bypass: true, name: "Mine" }));
    const out = resetNode(s, 2);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    const next = film(applyPatch(s, out.patch));
    expect(next.film).toEqual({ stock: "kodak_portra_400", exposure_ev: 0 });
    expect(next.strength).toBe(100);
    expect(next.grain).toEqual({ amount: 100, size: 100, seed: 7 });
    expect([next.bypass, next.name]).toEqual([true, "Mine"]);
    expect(resetNode(applyPatch(withFilm(), {}), 2)).toMatchObject({ ok: true, patch: {} });
  });
});
