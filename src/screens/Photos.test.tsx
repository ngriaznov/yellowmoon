import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openPhotos, openTab, renderApp, tools } from "../test/harness";
import type { MockOptions } from "../api.mock";
import type { Harness } from "../test/harness";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

/** The mockup's state: photo 4 current, photos 3 to 5 picked. */
const MOCKUP = { current: 3, picked: [2, 3, 4] };

async function setup(options: MockOptions = MOCKUP): Promise<Harness> {
  h = renderApp(options);
  await openPhotos(h);
  return h;
}

const lookOf = (): string | undefined => h?.store.currentSettings()?.values.look;

describe("the header", () => {
  it("shows the folder, its date and count, and the actions", async () => {
    await setup();
    expect(screen.getByText("Lisbon")).toBeInTheDocument();
    expect(screen.getByText("2026-10-04 · 12 photos")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Before" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Crop" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Pick white" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Fit/ })).toHaveAttribute("aria-haspopup", "menu");
  });

  it("shows the render status from render_stats", async () => {
    await setup();
    expect(await screen.findByText("Neural · 42 ms · Display P3")).toBeInTheDocument();
  });

  it("says Export, or Export N with several photos picked", async () => {
    await setup();
    expect(screen.getByRole("button", { name: /Export 3/ })).toBeInTheDocument();
  });

  it("says plain Export with one photo or none picked", async () => {
    await setup({ current: 0, picked: [] });
    expect(screen.getByRole("button", { name: /^Export\s?⌘E$/ })).toBeInTheDocument();
  });

  it("keeps the update cell hidden while there is nothing to update", async () => {
    await setup();
    expect(screen.queryByText(/Update/)).toBeNull();
  });
});

describe("the photo and the filmstrip", () => {
  it("shows the current photo with the look chip and the camera line", async () => {
    await setup();
    const photo = screen.getByRole("main", { name: "Photo" });
    expect(within(photo).getByRole("img", { name: "Photo DSC04204, Neutral · 100%" })).toBeInTheDocument();
    expect(within(photo).getByText("Neutral · 100%")).toBeInTheDocument();
    expect(within(photo).getByText("Sony A1 · DSC04204 · ISO 400 · 1/125 · f/2.8")).toBeInTheDocument();
  });

  it("shows the position and the picked count as one line", async () => {
    await setup();
    const strip = screen.getByRole("navigation", { name: "Photos" });
    expect(within(strip).getByText("4 of 12 · 3 picked")).toBeInTheDocument();
    expect(strip).toHaveAccessibleDescription("4 of 12 · 3 picked");
    expect(within(strip).getAllByRole("button")).toHaveLength(12);
    expect(within(strip).getByRole("button", { name: "Show DSC04204, picked" })).toHaveAttribute("aria-current", "true");
  });

  it("selects a photo by click and loads its settings", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: "Show DSC04206" }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).toBe("DSC04206.ARW"));
    expect(screen.getByText("6 of 12 · 3 picked")).toBeInTheDocument();
    await waitFor(() => expect(h!.store.currentSettings()?.photo_id).toBe(6));
  });

  it("⌘-click toggles the pick and P does too", async () => {
    await setup();
    const thumb = screen.getByRole("button", { name: "Show DSC04206" });
    fireEvent.click(thumb, { metaKey: true });
    await waitFor(() => expect(screen.getByText("4 of 12 · 4 picked")).toBeInTheDocument());
    await h!.user.keyboard("p");
    await waitFor(() => expect(screen.getByText("4 of 12 · 3 picked")).toBeInTheDocument());
    // P toggles the photo on screen (4, which was picked), ⌘-click toggled 6.
    expect(h!.store.getState().session?.picked).toEqual([3, 5, 6]);
  });

  it("← and → move between photos", async () => {
    await setup();
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(screen.getByText("5 of 12 · 3 picked")).toBeInTheDocument());
    await h!.user.keyboard("{ArrowLeft}{ArrowLeft}");
    await waitFor(() => expect(screen.getByText("3 of 12 · 3 picked")).toBeInTheDocument());
  });

  it("⌘A picks every photo", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}a{/Meta}");
    await waitFor(() => expect(screen.getByText("4 of 12 · 12 picked")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /Export 12/ })).toBeInTheDocument();
  });

  it("marks a file that can't be opened and says why over the photo", async () => {
    await setup({ current: 0, picked: [] });
    h!.unmount();
    h = renderApp({ current: 1, picked: [], skipped: { DSC04202: "This file is from a camera mode we can't read yet." } });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    await waitFor(() => expect(screen.getByRole("button", { name: /Open photos/ })).not.toHaveAttribute("aria-disabled"));
    await h.user.click(screen.getByRole("button", { name: /Open photos/ }));
    // The current photo is the first one that is ready, so go to the broken one by hand.
    await h.user.click(await screen.findByRole("button", { name: "Show DSC04202, can't open" }));
    expect(await screen.findByText("This file can't be opened.")).toBeInTheDocument();
    expect(screen.getByText("This file is from a camera mode we can't read yet.")).toBeInTheDocument();
    expect(screen.getByText("Can't open")).toBeInTheDocument();
    // Nothing to adjust on it.
    expect(within(tools()).queryByRole("region", { name: "Automatic" })).toBeNull();
  });
});

describe("Before, zoom and the viewport", () => {
  it("Before flips the photo to the camera default, with its label, and hides the look chip", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: "Before" }));
    expect(screen.getByRole("button", { name: "Before" })).toHaveAttribute("aria-pressed", "true");
    const photo = screen.getByRole("main", { name: "Photo" });
    expect(within(photo).getByText("Before · camera default")).toBeInTheDocument();
    expect(within(photo).queryByText("Neutral · 100%")).toBeNull();
    expect(h!.api.viewport.before).toBe(true);
    await h!.user.keyboard("\\");
    await waitFor(() => expect(screen.getByRole("button", { name: "Before" })).toHaveAttribute("aria-pressed", "false"));
    expect(h!.api.viewport.before).toBe(false);
  });

  it("sends the photo cell's rectangle to the core", async () => {
    await setup();
    await waitFor(() => expect(h!.api.viewport.rect).not.toBeNull());
  });

  it("Z toggles Fit and 100 %, and the menu offers Fit, 50 %, 100 % and 200 %", async () => {
    await setup();
    await h!.user.keyboard("z");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 100 }));
    expect(screen.getByRole("button", { name: /^100%/ })).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: /^100%/ }));
    const menu = screen.getByRole("menu", { name: "Zoom" });
    expect(within(menu).getAllByRole("menuitemradio").map((i) => i.textContent)).toEqual(["Fit", "50%", "100%", "200%"]);
    await h!.user.click(within(menu).getByRole("menuitemradio", { name: "200%" }));
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 200 }));
    await h!.user.keyboard("z");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "fit" }));
  });

  it("⌘+ and ⌘− step through the zoom levels", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}={/Meta}");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 50 }));
    await h!.user.keyboard("{Meta>}={/Meta}{Meta>}={/Meta}{Meta>}={/Meta}");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 200 }));
    await h!.user.keyboard("{Meta>}-{/Meta}");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 100 }));
  });

  it("a double click on the photo toggles the zoom through the core", async () => {
    await setup();
    const photo = screen.getByRole("main", { name: "Photo" });
    fireEvent.doubleClick(photo, { clientX: 20, clientY: 10 });
    await waitFor(() => expect(h!.api.viewport.inputs.some((i) => i.kind === "toggle_zoom")).toBe(true));
    expect(screen.getByRole("button", { name: /^100%/ })).toBeInTheDocument();
  });

  it("reads a pixel under the pointer while ⌥ is held", async () => {
    await setup();
    const photo = screen.getByRole("main", { name: "Photo" });
    photo.getBoundingClientRect = () => ({ left: 0, top: 0, width: 1000, height: 666, right: 1000, bottom: 666, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    fireEvent.pointerMove(photo, { clientX: 500, clientY: 300, altKey: true });
    expect(await within(photo).findByText(/^R \d+ G \d+ B \d+$/)).toBeInTheDocument();
    fireEvent.pointerMove(photo, { clientX: 500, clientY: 300, altKey: false });
    await waitFor(() => expect(within(photo).queryByText(/^R \d+ G \d+ B \d+$/)).toBeNull());
  });
});

describe("the tabs", () => {
  it("has six tabs in order, the Look tab first", async () => {
    await setup();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Look", "Light", "Colour", "Detail", "Curve", "RAW"]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(within(tools()).getByRole("heading", { level: 2, name: "Look" })).toBeInTheDocument();
  });

  it("⌘1 to ⌘6 switch tabs and each shows its title block and cards", async () => {
    await setup();
    const expected: Array<[string, string, string[]]> = [
      ["2", "Light", ["Exposure", "High dynamic range"]],
      ["3", "Colour", ["White balance", "Colour balance"]],
      ["4", "Detail", ["Sharpening", "Noise reduction"]],
      ["5", "Curve", ["Curve"]],
      ["6", "RAW", ["Camera", "Decode", "Range"]],
      ["1", "Look", ["Automatic", "Look"]],
    ];
    for (const [key, title, cards] of expected) {
      await h!.user.keyboard(`{Meta>}${key}{/Meta}`);
      await waitFor(() => expect(within(tools()).getByRole("heading", { level: 2, name: title })).toBeInTheDocument());
      for (const card of cards) expect(await within(tools()).findByRole("region", { name: card })).toBeInTheDocument();
    }
  });

  it("the arrows of the Automatic card open the tab that owns the value", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: /^White balance .* Open Colour$/ }));
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
  });
});

describe("the Look tab", () => {
  it("shows the Automatic card with the resolved values and AUTO tags", async () => {
    await setup();
    const card = within(screen.getByRole("region", { name: "Automatic" }));
    expect(card.getByRole("button", { name: "Exposure +0.30 EV, Auto. Open Light" })).toBeInTheDocument();
    expect(card.getByRole("button", { name: "White balance 5450 K · tint 0, Auto. Open Colour" })).toBeInTheDocument();
    expect(card.getByRole("button", { name: "Noise Luminance 30, Auto. Open Detail" })).toBeInTheDocument();
    expect(card.getByRole("button", { name: "Camera A1 · A7R IV profile, Auto. Open RAW" })).toBeInTheDocument();
  });

  it("flips a row to EDITED when its value is changed, and Re-run brings AUTO back", async () => {
    await setup();
    await h!.store.update({ exposure: { kind: "manual", ev: 1.5 } });
    const card = within(screen.getByRole("region", { name: "Automatic" }));
    expect(await card.findByRole("button", { name: "Exposure +1.50 EV, Edited. Open Light" })).toBeInTheDocument();
    await h!.user.click(card.getByRole("button", { name: "Re-run" }));
    expect(await card.findByRole("button", { name: "Exposure +0.30 EV, Auto. Open Light" })).toBeInTheDocument();
  });

  it("shows the Look card: intensity, Apply to with counts, search and the built-in and ARRI groups", async () => {
    await setup();
    const card = within(screen.getByRole("region", { name: "Look" }));
    expect(card.getByRole("slider", { name: "Intensity" })).toBeInTheDocument();
    expect(card.getByRole("textbox", { name: "Intensity value" })).toHaveValue("100%");
    const apply = card.getByRole("radiogroup", { name: "Apply look to" });
    expect(within(apply).getAllByRole("radio").map((r) => r.textContent)).toEqual(["This photo", "Picked · 3", "All · 12"]);
    // The owner's decision of 2026-10-20: a look is for the photo on screen until the person says otherwise.
    expect(within(apply).getByRole("radio", { name: "This photo" })).toHaveAttribute("aria-checked", "true");
    expect(card.getByText("Built in")).toBeInTheDocument();
    expect(card.getByText("ARRI looks · LogC4")).toBeInTheDocument();
    expect(card.getAllByRole("button", { name: / look, key \d/ })).toHaveLength(9);
    expect(card.getByRole("button", { name: "Neutral look, key 1" })).toHaveAttribute("aria-pressed", "true");
    expect(card.getByRole("button", { name: "Film warm look, key 3" })).toBeInTheDocument();
    expect(card.getByRole("button", { name: "Monochrome look, key 9" })).toBeInTheDocument();
  });

  it("tags the default look and loads the tile thumbnails", async () => {
    await setup();
    const tile = screen.getByRole("button", { name: "Neutral look, key 1" });
    expect(within(tile).getByText("Default")).toBeInTheDocument();
    await waitFor(() => expect(tile.querySelector("img")?.getAttribute("src")).toMatch(/^(data:image\/png|blob:)/));
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img[src]").length).toBe(9));
  });

  it("draws a tile again when the request it came back to was cancelled (photo A, B, then A before A's tile was made)", async () => {
    h = renderApp(MOCKUP);
    // The first tile's request stays pending; the core cancels it when the person moves to another photo.
    let cancel: (error: unknown) => void = () => {};
    const ask = vi.spyOn(h.api, "lookThumbnail").mockImplementationOnce(() => new Promise<string>((_, no) => (cancel = no)));
    await openPhotos(h);
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img[src]").length).toBe(8));
    await h.user.click(screen.getByRole("button", { name: "Show DSC04206" }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).toBe("DSC04206.ARW"));
    await h.user.click(screen.getByRole("button", { name: /^Show DSC04204/ }));
    await waitFor(() => expect(h!.store.currentPhoto()?.file_name).toBe("DSC04204.ARW"));
    const neutral = (): number => ask.mock.calls.filter(([photo, look]) => photo === 4 && look === "builtin:neutral").length;
    // The tile that came back was handed the request that is still pending: nothing was asked of the core.
    expect(neutral()).toBe(1);
    cancel({ scope: "render", message: "The render was cancelled." });
    await waitFor(() => expect(neutral()).toBe(2));
    await waitFor(() => expect(screen.getByRole("button", { name: "Neutral look, key 1" }).querySelector("img")?.getAttribute("src")).toMatch(/^(data:image\/png|blob:)/));
  });

  it("asks for a new picture when a tile's does not load (the shell dropped the tiles of that photo)", async () => {
    await setup();
    const ask = vi.spyOn(h!.api, "lookThumbnail");
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img[src]").length).toBe(9));
    const img = screen.getByRole("button", { name: "Neutral look, key 1" }).querySelector("img")!;
    const before = img.getAttribute("src");
    fireEvent.error(img);
    await waitFor(() => expect(ask).toHaveBeenCalledTimes(1));
    expect(ask).toHaveBeenCalledWith(4, "builtin:neutral");
    await waitFor(() => {
      const now = screen.getByRole("button", { name: "Neutral look, key 1" }).querySelector("img")?.getAttribute("src");
      expect(now).toMatch(/^(data:image\/png|blob:)/);
      expect(now).not.toBe(before);
    });
    // The other tiles were not asked for again.
    expect(ask).toHaveBeenCalledTimes(1);
  });

  it("filters by name and counts the matches in each group", async () => {
    await setup();
    await h!.user.type(screen.getByRole("searchbox", { name: "Search looks" }), "film");
    expect(screen.getByText("2 of 7")).toBeInTheDocument();
    expect(screen.queryByText("Built in")).toBeNull();
    expect(screen.getAllByRole("button", { name: /look, key/ })).toHaveLength(2);
    await h!.user.clear(screen.getByRole("searchbox", { name: "Search looks" }));
    await h!.user.type(screen.getByRole("searchbox", { name: "Search looks" }), "zzz");
    expect(screen.getByText("No looks match “zzz”.")).toBeInTheDocument();
  });

  it("picking a look changes the chip and keeps the look to this photo, which is where Apply to starts", async () => {
    await setup();
    const apply = vi.spyOn(h!.api, "applyLook");
    await h!.user.click(screen.getByRole("button", { name: "Teal and orange look, key 6" }));
    await waitFor(() => expect(lookOf()).toBe("arri:3000:teal-and-orange"));
    expect(within(screen.getByRole("main", { name: "Photo" })).getByText("Teal and orange · 100%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Teal and orange look, key 6" })).toHaveAttribute("aria-pressed", "true");
    expect(apply).not.toHaveBeenCalled();
  });

  it("copies a look to the other photos when Apply to says All", async () => {
    await setup();
    await h!.user.click(screen.getByRole("radio", { name: "All · 12" }));
    const apply = vi.spyOn(h!.api, "applyLook");
    await h!.user.click(screen.getByRole("button", { name: "Teal and orange look, key 6" }));
    await waitFor(() => expect(lookOf()).toBe("arri:3000:teal-and-orange"));
    await waitFor(() => expect(apply).toHaveBeenCalledWith("all", "arri:3000:teal-and-orange", 100));
  });

  it("keys 1 to 9 pick the looks", async () => {
    await setup();
    await h!.user.keyboard("3");
    await waitFor(() => expect(lookOf()).toBe("arri:3000:film-warm"));
    await h!.user.keyboard("2");
    await waitFor(() => expect(lookOf()).toBe("builtin:signature"));
    await h!.user.keyboard("9");
    await waitFor(() => expect(lookOf()).toBe("arri:3000:monochrome"));
  });

  it("keys 1 to 9 do nothing on another tab, or while typing in the search box", async () => {
    await setup();
    await openTab(h!, "Light");
    await h!.user.keyboard("3");
    expect(lookOf()).toBe("builtin:neutral");
    await openTab(h!, "Look");
    await h!.user.click(screen.getByRole("searchbox", { name: "Search looks" }));
    await h!.user.keyboard("3");
    expect(lookOf()).toBe("builtin:neutral");
    expect(screen.getByRole("searchbox", { name: "Search looks" })).toHaveValue("3");
  });

  it("offers Make default for a look that isn't the default, and makes it", async () => {
    await setup();
    expect(screen.queryByRole("button", { name: "Make default" })).toBeNull();
    await h!.user.click(screen.getByRole("button", { name: "Signature look, key 2" }));
    await h!.user.click(await screen.findByRole("button", { name: "Make default" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Make default" })).toBeNull());
    expect(within(screen.getByRole("button", { name: "Signature look, key 2" })).getByText("Default")).toBeInTheDocument();
    expect(within(screen.getByRole("button", { name: "Neutral look, key 1" })).queryByText("Default")).toBeNull();
  });

  it("Reset goes back to the default look at 100 %", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: "Night blue look, key 8" }));
    await waitFor(() => expect(lookOf()).toBe("arri:3000:night-blue"));
    await h!.user.click(within(screen.getByRole("region", { name: "Look" })).getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(lookOf()).toBe("builtin:neutral"));
    expect(h!.store.currentSettings()?.values.look_intensity).toBe(100);
  });

  it("shows a look whose file is missing as disabled, with the reason, and won't pick it", async () => {
    await setup({ ...MOCKUP, missingLooks: ["film-warm"] });
    const tile = screen.getByRole("button", { name: /Film warm look, key 3, file missing or changed/ });
    expect(tile).toHaveAttribute("aria-disabled", "true");
    expect(within(tile).getByText("File missing or changed. Import again.")).toBeInTheDocument();
    await h!.user.click(tile);
    await h!.user.keyboard("3");
    expect(lookOf()).toBe("builtin:neutral");
  });

  it("does not ask for the thumbnail of a look whose file is missing, and loads the others", async () => {
    h = renderApp({ ...MOCKUP, missingLooks: ["film-warm"] });
    const ask = vi.spyOn(h.api, "lookThumbnail");
    await openPhotos(h);
    await waitFor(() => expect(document.querySelectorAll(".ym-look-tile img[src]").length).toBe(8));
    expect(ask).toHaveBeenCalledTimes(8);
    expect(ask.mock.calls.map(([, look]) => look)).not.toContain("arri:3000:film-warm");
  });

  it("offers the wizard in place of the ARRI looks when none are imported", async () => {
    await setup({ ...MOCKUP, looksImported: false });
    expect(screen.queryByText(/^ARRI looks · /)).toBeNull();
    expect(screen.getByText("The free ARRI Look Library is not imported yet.")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /look, key/ })).toHaveLength(2);
    await h!.user.click(screen.getByRole("button", { name: "Import ARRI looks →" }));
    expect(screen.getByRole("heading", { name: "Choose what to import" })).toBeInTheDocument();
  });

  it("opens the wizard from the link under the grid, and Back returns to the photos", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: "Import ARRI looks →" }));
    expect(screen.getByRole("heading", { name: "Choose what to import" })).toBeInTheDocument();
    await h!.user.click(screen.getByRole("button", { name: "Cancel" }));
    // Back goes to where it came from.
    expect(await screen.findByRole("tab", { name: "Look" })).toBeInTheDocument();
  });
});

describe("undo, redo and closing", () => {
  it("⌘Z and ⇧⌘Z undo and redo a change for this photo", async () => {
    await setup();
    await openTab(h!, "Light");
    await h!.store.update({ contrast: manual(30) });
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+30"));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Contrast value" })).toHaveValue("0"));
    await h!.user.keyboard("{Meta>}{Shift>}z{/Shift}{/Meta}");
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Contrast value" })).toHaveValue("+30"));
  });

  it("⌘W closes the session and shows Start", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}w{/Meta}");
    expect(await screen.findByRole("group", { name: "Drop files or a folder" })).toBeInTheDocument();
    expect((await h!.api.sessionState()).photos).toHaveLength(0);
  });

  it("⌘E opens the export panel, and Esc closes it", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}e{/Meta}");
    expect(await screen.findByRole("complementary", { name: "Export" })).toBeInTheDocument();
    await h!.user.keyboard("{Escape}");
    expect(await screen.findByRole("complementary", { name: "Tools" })).toBeInTheDocument();
  });

  it("shows the update cell when the updater has news", async () => {
    const { updater } = await import("../updater");
    await setup();
    act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
    expect(await screen.findByText("Update ready")).toBeInTheDocument();
    act(() => updater.applyExternal({ phase: "idle" }));
    await waitFor(() => expect(screen.queryByText("Update ready")).toBeNull());
  });
});

describe("the RAW tab's camera card", () => {
  const camera = () => within(screen.getByRole("region", { name: "Camera" }));

  it("says when a body shares its sensor with another and uses that profile", async () => {
    await setup({ ...MOCKUP, photoOverrides: { DSC04204: { camera: { body: "Sony A7 IV", model: "ILCE-7M4", profile: "A7 III", match_kind: { kind: "sensor_family", via: "Sony A7 III" } } } } });
    await openTab(h!, "RAW");
    expect(camera().getByText("Sony A7 IV (ILCE-7M4)")).toBeInTheDocument();
    expect(camera().getByText("The A7 IV shares its sensor with the A7 III, so it uses that profile.")).toBeInTheDocument();
    expect(camera().getByText("Stand-in")).toBeInTheDocument();
  });

  it("has no STAND-IN tag for a body with its own measured profile", async () => {
    await setup({ ...MOCKUP, photoOverrides: { DSC04204: { camera: { body: "Sony A7R IV", model: "ILCE-7RM4", profile: "A7R IV", match_kind: { kind: "exact" } } } } });
    await openTab(h!, "RAW");
    expect(camera().queryByText("Stand-in")).toBeNull();
    expect(camera().getByText("The A7R IV has its own measured profile.")).toBeInTheDocument();
    await openTab(h!, "Look");
    expect(screen.getByRole("button", { name: "Camera A7R IV · measured profile, Auto. Open RAW" })).toBeInTheDocument();
  });

  it("adds the REDUCED tag and the note for a Lossless M or S file", async () => {
    await setup({ ...MOCKUP, photoOverrides: { DSC04204: { reduced: true } } });
    await openTab(h!, "RAW");
    expect(camera().getByText("Reduced")).toBeInTheDocument();
    expect(camera().getByText(/^This file was saved in the camera as Lossless M or S\./)).toBeInTheDocument();
    expect(camera().getByText("Lossless M or S")).toBeInTheDocument();
  });
});
