/**
 * The store's part in the Film node's pickers (`docs/24` section 4.4): the list of stocks is read once, and the stock tiles are asked for one
 * at a time, kept per photo, node and settings, dropped when the settings or the photo change, and made again when an address stops answering.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import type { MockApi } from "../api.mock";
import type { NodeLink, SettingsPatch } from "../api/types";
import { isSuperseded } from "./requestLimiter";
import { createStore } from "./store";
import type { AppStore } from "./store";

const stops: Array<() => void> = [];
afterEach(() => {
  for (const stop of stops.splice(0)) stop();
});

const picture = (from: number | "input", to: number | "output"): NodeLink => ({ from: { node: from, port: "picture" }, to: { node: to, port: "picture" } });
const WITH_FILM: SettingsPatch = { nodes: { "2": { kind: "film", name: "Film", pos: [1, 0] } }, links: [picture("input", 1), picture(1, 2), picture(2, "output")] };

async function opened(): Promise<{ api: MockApi; store: AppStore }> {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, current: 3, initialSettings: { DSC04204: WITH_FILM } });
  const store = createStore(api, 0);
  stops.push(store.connect());
  await store.load();
  await store.openDialog("files");
  api.flush();
  await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
  await store.setCurrent(4);
  await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
  return { api, store };
}

describe("the Film node's list", () => {
  it("is read once", async () => {
    const { api, store } = await opened();
    const ask = vi.spyOn(api, "filmCatalogue");
    const [a, b] = await Promise.all([store.filmCatalogue(), store.filmCatalogue()]);
    expect(a).toBe(b);
    await store.filmCatalogue();
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it("is read again after a failure", async () => {
    const { api, store } = await opened();
    api.failNext("filmCatalogue", { scope: "render", message: "Yellowmoon's film data is missing. Reinstall the app." });
    await expect(store.filmCatalogue()).rejects.toMatchObject({ message: "Yellowmoon's film data is missing. Reinstall the app." });
    await expect(store.filmCatalogue()).resolves.toMatchObject({ stocks: expect.any(Array) });
  });
});

describe("the stock tiles", () => {
  it("are asked for once per stock and kept for the same settings", async () => {
    const { api, store } = await opened();
    const ask = vi.spyOn(api, "filmTile");
    const first = await store.filmTile(4, 2, "kodak_ektar_100", "k1");
    expect(await store.filmTile(4, 2, "kodak_ektar_100", "k1")).toBe(first);
    expect(ask).toHaveBeenCalledTimes(1);
    expect(ask).toHaveBeenCalledWith(4, 2, "kodak_ektar_100");
  });

  it("are asked for one at a time, in the order they were asked for", async () => {
    const { api, store } = await opened();
    const order: string[] = [];
    let running = 0;
    let most = 0;
    vi.spyOn(api, "filmTile").mockImplementation(async (_photo, _node, stock) => {
      running += 1;
      most = Math.max(most, running);
      order.push(stock);
      await new Promise((resolve) => setTimeout(resolve, 2));
      running -= 1;
      return `blob:${stock}`;
    });
    const stocks = ["kodak_ektar_100", "kodak_gold_200", "kodak_portra_160", "fujifilm_c200"];
    await Promise.all(stocks.map((s) => store.filmTile(4, 2, s, "k1")));
    expect(most).toBe(1);
    expect(order).toEqual(stocks);
  });

  it("that wait are dropped when the settings change, and are not a failure", async () => {
    const { api, store } = await opened();
    vi.spyOn(api, "filmTile").mockImplementation(() => new Promise<string>((resolve) => setTimeout(() => resolve("blob:x"), 5)));
    const running = store.filmTile(4, 2, "kodak_ektar_100", "k1");
    const waiting = store.filmTile(4, 2, "kodak_gold_200", "k1");
    const newer = store.filmTile(4, 2, "kodak_gold_200", "k2");
    await expect(waiting).rejects.toSatisfy(isSuperseded);
    await expect(running).resolves.toBe("blob:x");
    await expect(newer).resolves.toBe("blob:x");
  });

  it("are made again for a picture that did not load", async () => {
    const { api, store } = await opened();
    const ask = vi.spyOn(api, "filmTile");
    const url = await store.filmTile(4, 2, "kodak_ektar_100", "k1");
    store.forgetFilmTile(4, 2, "kodak_ektar_100", "k1", url);
    const again = await store.filmTile(4, 2, "kodak_ektar_100", "k1");
    expect(again).not.toBe(url);
    expect(ask).toHaveBeenCalledTimes(2);
  });

  it("are not kept when the request failed", async () => {
    const { api, store } = await opened();
    api.failNext("filmTile", { scope: "render", message: "The render was cancelled." });
    await expect(store.filmTile(4, 2, "kodak_ektar_100", "k1")).rejects.toMatchObject({ scope: "render" });
    await expect(store.filmTile(4, 2, "kodak_ektar_100", "k1")).resolves.toMatch(/^blob:/);
  });
});
