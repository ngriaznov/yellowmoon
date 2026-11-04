import { afterEach, describe, expect, it } from "vitest";
import { createMockApi } from "./api.mock";
import type { MockApi } from "./api.mock";
import type { DevelopSettings, FilmNode, SettingsPatch } from "./api/types";
import { FILM_CREDIT } from "./fixtures/filmCredit";
import { MOCK_FILM_PAPERS, MOCK_FILM_STOCKS } from "./fixtures/film";
import { nodeOf } from "./state/settingsGraph";

let api: MockApi;
afterEach(() => api?.reset());

const picture = (from: number | "input", to: number | "output") => ({ from: { node: from, port: "picture" as const }, to: { node: to, port: "picture" as const } });
const WITH_FILM: SettingsPatch = {
  nodes: { "2": { kind: "film", name: "Film", pos: [1, 0] } },
  links: [picture("input", 1), picture(1, 2), picture(2, "output")],
};

async function open(initial: SettingsPatch = WITH_FILM): Promise<{ id: number; settings: () => Promise<DevelopSettings> }> {
  api = createMockApi({ speed: 1000, initialSettings: { DSC04201: initial } });
  const session = await api.openPaths(["/photos/DSC04201.ARW"]);
  const id = session.photos[0]!.id;
  await api.setCurrent(id);
  api.flush();
  return { id, settings: async () => (await api.getSettings(id)).settings };
}
const filmOf = (settings: DevelopSettings): FilmNode => nodeOf(settings, 2) as unknown as FilmNode;

describe("the mock's Film node", () => {
  it("answers the catalogue as the core does: 20 films in three kinds, 6 papers, the credit lines", async () => {
    api = createMockApi({ speed: 1000 });
    const c = await api.filmCatalogue();
    expect(c.stocks).toHaveLength(20);
    expect(c.papers).toHaveLength(6);
    expect(c.stocks.filter((s) => s.kind === "slide").every((s) => s.matching_paper === null)).toBe(true);
    expect(c.stocks.find((s) => s.id === "kodak_vision3_500t")?.matching_paper).toBe("kodak_portra_endura");
    expect(c.credit).toBe(FILM_CREDIT);
    expect(c.disclaimer).toBe("Modelled on published data. Not affiliated with Kodak or Fujifilm.");
    expect(c.trademarks).toBe("Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners.");
    expect(c.papers.map((p) => p.id)).not.toContain("kodak_2383");
    expect(MOCK_FILM_STOCKS).toHaveLength(20);
    expect(MOCK_FILM_PAPERS).toHaveLength(6);
  });

  it("clamps the numbers of a Film node to their ranges, the lights to halves", async () => {
    const { id, settings } = await open();
    await api.updateSettings(id, {
      nodes: { "2": { film: { exposure_ev: 9 }, print: { exposure_ev: -9, lights: { y: 12.26, c: -80 } }, halation: { amount: 999, size: 1 }, grain: { amount: -4, size: 999 }, strength: 140 } },
    });
    const film = filmOf(await settings());
    expect(film.film.exposure_ev).toBe(4);
    expect(film.print.exposure_ev).toBe(-3);
    expect(film.print.lights).toEqual({ c: -50, m: 0, y: 12.5 });
    expect(film.halation).toEqual({ amount: 300, size: 50 });
    expect(film.grain).toMatchObject({ amount: 0, size: 200, seed: 0 });
    expect(film.strength).toBe(100);
  });

  it("refuses a stock or a paper it does not have, in the core's words", async () => {
    const { id } = await open();
    await expect(api.updateSettings(id, { nodes: { "2": { film: { stock: "kodak_nope" } } } })).rejects.toMatchObject({
      message: 'Unknown film stock "kodak_nope". Run "ym film list" to see the stocks.',
    });
    await expect(api.updateSettings(id, { nodes: { "2": { print: { paper: "kodak_2383" } } } })).rejects.toMatchObject({
      message: 'Unknown film paper "kodak_2383". Run "ym film list" to see the papers.',
    });
  });

  it("makes a tile for each stock of a Film node, a new address every time, and refuses a node that is not one", async () => {
    const { id } = await open();
    const a = await api.filmTile(id, 2, "kodak_ektar_100");
    const b = await api.filmTile(id, 2, "kodak_ektar_100");
    expect(a).toMatch(/^blob:/);
    expect(b).not.toBe(a);
    await expect(api.filmTile(id, 1, "kodak_ektar_100")).rejects.toMatchObject({ scope: "render", message: "This photo has no Film node 1." });
    await expect(api.filmTile(id, 2, "nope")).rejects.toMatchObject({ scope: "render" });
  });

  it("a stock patch is kept", async () => {
    const { id, settings } = await open();
    await api.updateSettings(id, { nodes: { "2": { film: { stock: "kodak_ektar_100" } } } });
    expect(filmOf(await settings()).film.stock).toBe("kodak_ektar_100");
  });
});
