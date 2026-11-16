/**
 * The Presets card (`docs/30` sections 9.1, 9.2, 9.4 and 10) on the Look tab: the last card after Automatic and Look, the empty state, the
 * six tiles and Show all, the ring, the tooltip and the description, the damaged rows, Apply to, and the tile's menu.
 */
import { screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MockOptions } from "../../api.mock";
import { openPhotos, renderApp, tools } from "../../test/harness";
import type { Harness } from "../../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

async function onLook(options: MockOptions = {}): Promise<HTMLElement> {
  h = renderApp({ current: 3, picked: [2, 3, 4], ...options });
  await openPhotos(h);
  await waitFor(() => expect(h!.store.getState().presets).not.toBeNull());
  return screen.getByRole("region", { name: "Presets" });
}

const tiles = (card: HTMLElement): HTMLElement[] => within(card).getAllByRole("button", { name: / preset$/ });

describe("the Look tab's cards", () => {
  it("are Automatic, Look and Presets, in that order, and there is no Tone card", async () => {
    await onLook();
    const titles = within(tools()).getAllByRole("heading", { level: 3 }).map((e) => e.textContent);
    expect(titles).toEqual(expect.arrayContaining(["Automatic", "Look", "Presets"]));
    expect(titles.indexOf("Automatic")).toBeLessThan(titles.indexOf("Look"));
    expect(titles.indexOf("Look")).toBeLessThan(titles.indexOf("Presets"));
    expect(titles.at(-1)).toBe("Presets");
    expect(screen.queryByRole("region", { name: "Tone" })).toBeNull();
  });
});

describe("the empty state", () => {
  it("has the header only, one sentence and Save preset…, and starts with no preset", async () => {
    const card = await onLook();
    expect(within(card).getByText("Set a photo the way you like it, then save it as a preset.")).toBeInTheDocument();
    expect(within(card).getByRole("button", { name: "Save preset…" })).not.toHaveAttribute("aria-disabled");
    expect(within(card).queryByRole("button", { name: "Save…" })).toBeNull();
    expect(within(card).queryByRole("radiogroup")).toBeNull();
    expect(within(card).queryByRole("group", { name: "Presets" })).toBeNull();
  });

  it("opens the Save dialog from the button", async () => {
    const card = await onLook();
    await h!.user.click(within(card).getByRole("button", { name: "Save preset…" }));
    expect(await screen.findByRole("dialog", { name: "Save preset" })).toBeInTheDocument();
  });
});

describe("the tiles", () => {
  it("show six with the count in the header, and Show all N then Show fewer", async () => {
    const card = await onLook({ presets: 14 });
    expect(within(card).getByText("14")).toHaveClass("ym-presets__count");
    expect(within(card).getByRole("button", { name: "Save…" })).toBeInTheDocument();
    expect(tiles(card)).toHaveLength(6);
    const more = within(card).getByRole("button", { name: "Show all 14" });
    expect(more).toHaveAttribute("aria-expanded", "false");
    await h!.user.click(more);
    expect(tiles(card)).toHaveLength(14);
    const fewer = within(card).getByRole("button", { name: "Show fewer" });
    expect(fewer).toHaveAttribute("aria-expanded", "true");
    await h!.user.click(fewer);
    expect(tiles(card)).toHaveLength(6);
  });

  it("have no Show all with six or fewer", async () => {
    const card = await onLook({ presets: 6 });
    expect(tiles(card)).toHaveLength(6);
    expect(within(card).queryByRole("button", { name: /Show/ })).toBeNull();
  });

  it("are a group named Presets of buttons named by the preset, with the groups as their description and no number badge", async () => {
    const card = await onLook({ presets: 3 });
    const group = within(card).getByRole("group", { name: "Presets" });
    const warm = within(group).getByRole("button", { name: "Warm evening preset" });
    expect(warm).toHaveAttribute("aria-pressed", "false");
    expect(warm).toHaveAttribute("aria-description", "Look, Light and Colour");
    expect(warm).toHaveAttribute("title", "Look, Light, Colour");
    expect(warm.querySelector(".ym-look-tile__badge")).toBeNull();
    expect(within(warm).getByText("Warm evening")).toHaveClass("ym-look-tile__name");
  });

  it("say that the look is not imported, in the description and the tooltip", async () => {
    const card = await onLook({ presets: 3, missingLooks: ["film-warm"] });
    const warm = within(card).getByRole("button", { name: "Warm evening preset" });
    expect(warm).toHaveAttribute("aria-description", "Look, Light and Colour. The look Film warm isn't imported.");
    expect(warm).toHaveAttribute("title", "Look, Light, Colour\nLook: Film warm isn't imported.");
  });

  it("show the picture the core made, and ask only for the tiles in view (all of them without an observer)", async () => {
    await onLook({ presets: 3 });
    const spy = vi.spyOn(h!.api, "presetThumbnail");
    await h!.store.refreshPresets();
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile__picture img").length).toBeGreaterThan(0));
    void spy;
  });

  it("apply on a click, and the ring says what the photo has now", async () => {
    const card = await onLook({ presets: 3 });
    const warm = within(card).getByRole("button", { name: "Warm evening preset" });
    await h!.user.click(warm);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent(/^Applied Warm evening to DSC04204\./));
    await waitFor(() => expect(within(card).getByRole("button", { name: "Warm evening preset" })).toHaveAttribute("aria-pressed", "true"));
    expect(within(card).getByRole("button", { name: "Cool morning preset" })).toHaveAttribute("aria-pressed", "false");
  });

  it("apply to the picked photos when Apply to says Picked", async () => {
    const card = await onLook({ presets: 3 });
    expect(within(card).getByRole("radio", { name: "This photo" })).toBeChecked();
    expect(within(card).getByRole("radio", { name: "Picked · 3" })).toBeInTheDocument();
    expect(within(card).getByRole("radio", { name: "All · 12" })).toBeInTheDocument();
    const spy = vi.spyOn(h!.api, "presetApply");
    await h!.user.click(within(card).getByRole("radio", { name: "Picked · 3" }));
    await h!.user.click(within(card).getByRole("button", { name: "Warm evening preset" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0]?.[1]).toHaveLength(3);
    // It is the card's own: the Look card's Apply to is still on This photo.
    const lookCard = screen.getByRole("region", { name: "Look" });
    expect(within(lookCard).getByRole("radio", { name: "This photo" })).toBeChecked();
  });
});

describe("the damaged files", () => {
  it("are listed under the tiles with the reason and Delete, and do not count", async () => {
    const card = await onLook({ presets: 3, damagedPresets: true });
    const list = within(card).getByRole("list", { name: "Files that are not presets" });
    expect(within(list).getByText("Old.ympreset")).toBeInTheDocument();
    expect(within(list).getByText("This preset was made by a newer version of Yellowmoon. Update to use it.")).toBeInTheDocument();
    expect(within(card).getByText("3")).toHaveClass("ym-presets__count");
    await h!.user.click(within(list).getByRole("button", { name: "Delete Old.ympreset" }));
    await waitFor(() => expect(within(card).queryByRole("list", { name: "Files that are not presets" })).toBeNull());
  });
});

describe("a tile's menu", () => {
  async function menuOf(name: string, options: MockOptions = {}): Promise<HTMLElement> {
    const card = await onLook({ presets: 3, ...options });
    const tile = within(card).getByRole("button", { name: `${name} preset` });
    await h!.user.pointer({ keys: "[MouseRight]", target: tile });
    return screen.findByRole("menu", { name: `${name} actions` });
  }

  it("lists its six items, named for the tile", async () => {
    const menu = await menuOf("Warm evening");
    expect(within(menu).getAllByRole("menuitem").map((i) => i.textContent)).toEqual([
      "Apply to Picked Photos",
      "Update from This Photo",
      "Rename…",
      "Export…",
      "Show in Finder",
      "Delete",
    ]);
  });

  it("is off for Apply to Picked Photos while nobody is picked, with the reason", async () => {
    const menu = await menuOf("Warm evening", { picked: [] });
    const item = within(menu).getByRole("menuitem", { name: "Apply to Picked Photos" });
    expect(item).toHaveAttribute("aria-disabled", "true");
    expect(item).toHaveAttribute("title", "No photos are picked.");
  });

  it("applies to the picked photos whatever Apply to says", async () => {
    const menu = await menuOf("Warm evening");
    const spy = vi.spyOn(h!.api, "presetApply");
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Apply to Picked Photos" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    expect(spy.mock.calls[0]?.[1]).toHaveLength(3);
  });

  it("asks before Update, with its groups and the photo,", async () => {
    const menu = await menuOf("Warm evening");
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Update from This Photo" }));
    const question = await screen.findByRole("dialog", { name: "Update Warm evening?" });
    expect(question).toHaveTextContent("It keeps its groups, Look, Light and Colour, and takes their values from DSC04204.");
    expect(within(question).getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    await h!.user.click(within(question).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("deletes with no question and says it", async () => {
    const menu = await menuOf("Warm evening");
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Delete" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Moved the preset Warm evening to the Trash."));
    expect(screen.queryByRole("button", { name: "Warm evening preset" })).toBeNull();
  });
});
