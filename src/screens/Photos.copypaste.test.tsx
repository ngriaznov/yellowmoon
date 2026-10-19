/**
 * "Add photos" in the header, "Copy settings" and "Paste" at the foot of the tool panel, their keys, and drops on the Photos screen
 * (the owner's request of 2026-10-16, `docs/11` §Photos and §Copy and paste settings).
 */
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openPhotos, renderApp, tools } from "../test/harness";
import type { Harness } from "../test/harness";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

/** The mockup's state: photo 4 current, photos 3 to 5 picked. */
async function setup(options = { current: 3, picked: [2, 3, 4] }): Promise<Harness> {
  h = renderApp(options);
  await openPhotos(h);
  return h;
}

const header = (): HTMLElement => screen.getByRole("navigation", { name: "Actions" });
const foot = (): HTMLElement => within(tools()).getByRole("group", { name: "Copy and paste settings" });
const copyButton = (): HTMLElement => within(foot()).getByRole("button", { name: "Copy settings" });
const pasteButton = (): HTMLElement => within(foot()).getByRole("button", { name: "Paste" });
const contrastOf = (id: number): Promise<number> => h!.api.getSettings(id).then((rs) => rs.values.contrast);

const dropOnWindow = (...names: string[]): void => {
  const data = { types: ["Files"], files: names.map((name) => ({ name })), dropEffect: "none" };
  act(() => {
    fireEvent.dragEnter(window, { dataTransfer: data });
    fireEvent.drop(window, { dataTransfer: data });
  });
};

describe("Add photos", () => {
  it("is a header button after Pick white and before the zoom menu, and says its key", async () => {
    await setup();
    const labels = within(header()).getAllByRole("button").map((b) => b.textContent);
    expect(labels.slice(0, 5)).toEqual(["Before", "Crop", "Pick white", "Add photos", expect.stringMatching(/^Fit/)]);
    expect(within(header()).getByRole("button", { name: "Add photos" })).toHaveAccessibleDescription("Add photos (⌘O)");
  });

  it("opens the system's dialog for files and adds what is chosen, keeping the current photo", async () => {
    await setup();
    h!.api.queueChoice("open", ["~/Pictures/Lisbon/DSC09001.ARW", "~/Pictures/Lisbon/DSC04201.ARW"]);
    const spy = vi.spyOn(h!.api, "openDialog");
    await h!.user.click(within(header()).getByRole("button", { name: "Add photos" }));
    expect(spy).toHaveBeenCalledWith("files");
    await waitFor(() => expect(h!.store.getState().session?.photos).toHaveLength(13));
    // The filmstrip has the new photo after the others, and the current photo is still DSC04204.
    const strip = screen.getByRole("navigation", { name: "Photos" });
    expect(within(strip).getAllByRole("button")).toHaveLength(13);
    expect(within(strip).getByRole("button", { name: /^Show DSC09001/ })).toBeInTheDocument();
    expect(within(strip).getByRole("button", { name: "Show DSC04204, picked" })).toHaveAttribute("aria-current", "true");
    expect(screen.getByText("4 of 13 · 3 picked")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Added 1 photo. 1 was already open.");
    // The Export panel counts the new photo.
    await h!.user.click(screen.getByRole("button", { name: /Export/ }));
    await screen.findByRole("heading", { level: 2, name: "Export" });
    expect(screen.getByRole("radio", { name: /All · 13/ })).toBeInTheDocument();
  });

  it("says so when every photo is open already", async () => {
    await setup();
    await h!.user.click(within(header()).getByRole("button", { name: "Add photos" }));
    expect(await screen.findByRole("status")).toHaveTextContent("These photos are already open.");
    expect(h!.store.getState().session?.photos).toHaveLength(12);
  });

  it("is off while a dialog is open, so it cannot ask twice", async () => {
    await setup();
    let finish: (value: null) => void = () => {};
    const spy = vi.spyOn(h!.api, "openDialog").mockReturnValue(new Promise((resolve) => (finish = resolve)));
    const button = within(header()).getByRole("button", { name: "Add photos" });
    await h!.user.click(button);
    await waitFor(() => expect(button).toHaveAttribute("aria-disabled", "true"));
    await h!.user.click(button);
    expect(spy).toHaveBeenCalledTimes(1);
    await act(async () => finish(null));
    await waitFor(() => expect(button).not.toHaveAttribute("aria-disabled"));
  });

  it("is not there while the Export panel shows, like the other actions", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: /Export/ }));
    await screen.findByRole("heading", { level: 2, name: "Export" });
    expect(within(header()).queryByRole("button", { name: "Add photos" })).toBeNull();
    expect(within(header()).getByRole("button", { name: "← Back to photos" })).toBeInTheDocument();
  });

  it("⌘O adds too, with the same notice", async () => {
    await setup();
    h!.api.queueChoice("open", ["~/Pictures/Lisbon/DSC09002.ARW"]);
    await h!.user.keyboard("{Meta>}o{/Meta}");
    await waitFor(() => expect(h!.store.getState().session?.photos).toHaveLength(13));
    expect(screen.getByRole("status")).toHaveTextContent("Added 1 photo.");
  });

  it("takes files dropped anywhere on the window, and frames the window while they are dragged over it", async () => {
    await setup();
    const screenRoot = document.querySelector("[data-screen=photos]")!;
    expect(screenRoot).not.toHaveAttribute("data-dragging");
    act(() => {
      fireEvent.dragEnter(window, { dataTransfer: { types: ["Files"], files: [] } });
    });
    expect(screenRoot).toHaveAttribute("data-dragging", "true");
    act(() => {
      fireEvent.dragLeave(window, { dataTransfer: { types: ["Files"], files: [] } });
    });
    expect(screenRoot).not.toHaveAttribute("data-dragging");

    const spy = vi.spyOn(h!.api, "openPaths");
    dropOnWindow("DSC09100.ARW", "notes.txt");
    await waitFor(() => expect(spy).toHaveBeenCalledWith(["DSC09100.ARW"]));
    await waitFor(() => expect(h!.store.getState().session?.photos).toHaveLength(13));
    expect(screen.getByRole("status")).toHaveTextContent("Added 1 photo.");
    expect(screenRoot).not.toHaveAttribute("data-dragging");
  });

  it("says so when a drop holds no ARW files, and opens nothing", async () => {
    await setup();
    const spy = vi.spyOn(h!.api, "openPaths");
    dropOnWindow("holiday.jpg");
    expect(await screen.findByRole("status")).toHaveTextContent("No ARW files in what you dropped.");
    expect(spy).not.toHaveBeenCalled();
  });

  it("leaves drops to a dialog that is open", async () => {
    await setup();
    await h!.user.keyboard("{Meta>},{/Meta}");
    await screen.findByRole("dialog");
    const spy = vi.spyOn(h!.api, "openPaths");
    dropOnWindow("DSC09100.ARW");
    expect(spy).not.toHaveBeenCalled();
  });
});

describe("Copy settings and Paste", () => {
  it("are a row at the foot of the tool panel, with their keys in the tooltips", async () => {
    await setup();
    expect(copyButton()).toHaveAccessibleDescription("Copy the settings of this photo (⇧⌘C)");
    // Nothing is copied yet: Paste says what to do first.
    expect(pasteButton()).toHaveAttribute("aria-disabled", "true");
    expect(pasteButton()).toHaveAccessibleDescription("Copy the settings of a photo first (⇧⌘C)");
    // The header has neither.
    expect(within(header()).queryByRole("button", { name: "Copy settings" })).toBeNull();
    expect(within(header()).queryByRole("button", { name: "Paste" })).toBeNull();
  });

  it("copy on one photo and paste on another: its settings follow, its crop does not, and Undo puts it back", async () => {
    await setup();
    await h!.store.update({ contrast: manual(25), exposure: { kind: "manual", ev: 0.9 }, orientation: 90 });
    await h!.user.click(copyButton());
    expect(await screen.findByRole("status")).toHaveTextContent("Copied the settings of DSC04204.");
    expect(pasteButton()).not.toHaveAttribute("aria-disabled");
    expect(pasteButton()).toHaveAccessibleDescription("Paste the settings of DSC04204 (⇧⌘V)");

    await h!.user.click(screen.getByRole("button", { name: "Show DSC04206" }));
    await waitFor(() => expect(h!.store.currentSettings()?.photo_id).toBe(6));
    await h!.store.update({ crop: { rect: { x: 0.2, y: 0.1, width: 0.5, height: 0.7 }, angle_deg: 0, aspect: "free" } });
    await h!.user.click(pasteButton());
    await waitFor(() => expect(h!.store.currentSettings()?.settings.contrast).toEqual(manual(25)));
    const pasted = h!.store.currentSettings()!;
    expect(pasted.settings.exposure).toEqual({ kind: "manual", ev: 0.9 });
    expect(pasted.settings.orientation).toBe(0);
    expect(pasted.settings.crop?.rect.width).toBe(0.5);
    expect(screen.getByRole("status")).toHaveTextContent("Pasted the settings of DSC04204.");
    // The sliders show it.
    await waitFor(() => expect(screen.getByRole("tab", { name: "Light" })).toBeInTheDocument());
    await h!.user.click(screen.getByRole("tab", { name: "Light" }));
    await waitFor(() => expect((screen.getByRole("textbox", { name: "Contrast value" }) as HTMLInputElement).value).toBe("+25"));

    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(h!.store.currentSettings()?.settings.contrast).toEqual({ kind: "auto" }));
    expect(h!.store.currentSettings()?.settings.crop?.rect.width).toBe(0.5);
    // The photo it came from kept its settings.
    expect(await contrastOf(4)).toBe(25);
  });

  it("are off for a photo that is not developed", async () => {
    h = renderApp({ current: 0, holdAt: 2 });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    await waitFor(() => expect(h!.store.getState().session?.photos.filter((p) => p.status.kind === "ready")).toHaveLength(2));
    await waitFor(() => expect(screen.getByRole("button", { name: /Open photos/ })).not.toHaveAttribute("aria-disabled"));
    await h.user.click(screen.getByRole("button", { name: /Open photos/ }));
    await screen.findByRole("tab", { name: "Look" });
    await h.store.setCurrent(9);
    await waitFor(() => expect(copyButton()).toHaveAttribute("aria-disabled", "true"));
    expect(pasteButton()).toHaveAttribute("aria-disabled", "true");
  });

  it("⇧⌘C copies and ⇧⌘V pastes, and plain ⌘C and ⌘V are not theirs", async () => {
    await setup();
    await h!.store.update({ contrast: manual(14) });
    await h!.user.keyboard("{Meta>}c{/Meta}");
    expect(h!.store.getState().copied).toBeNull();
    await h!.user.keyboard("{Meta>}{Shift>}c{/Shift}{/Meta}");
    await waitFor(() => expect(h!.store.getState().copied?.source).toBe("DSC04204"));
    await h!.store.setCurrent(7);
    await waitFor(() => expect(h!.store.currentSettings()?.photo_id).toBe(7));
    await h!.user.keyboard("{Meta>}v{/Meta}");
    expect(await contrastOf(7)).toBe(0);
    await h!.user.keyboard("{Meta>}{Shift>}v{/Shift}{/Meta}");
    await waitFor(async () => expect(await contrastOf(7)).toBe(14));
  });

  it("⌥⇧⌘V pastes onto every picked photo", async () => {
    await setup();
    await h!.store.update({ contrast: manual(18) });
    await h!.user.keyboard("{Meta>}{Shift>}c{/Shift}{/Meta}");
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    await h!.user.keyboard("{Meta>}{Alt>}{Shift>}v{/Shift}{/Alt}{/Meta}");
    await waitFor(async () => expect(await contrastOf(3)).toBe(18));
    expect(await contrastOf(5)).toBe(18);
    expect(await contrastOf(1)).toBe(0);
    expect(screen.getByRole("status")).toHaveTextContent("Pasted the settings of DSC04204 to 3 photos.");
  });

  it("leave the keys to a text box that has focus", async () => {
    await setup();
    const search = screen.getByRole("searchbox", { name: "Search looks" });
    await h!.user.click(search);
    await h!.user.keyboard("{Meta>}{Shift>}c{/Shift}{/Meta}");
    expect(h!.store.getState().copied).toBeNull();
    // Copy something, then try to paste from the text box.
    await h!.user.click(copyButton());
    await waitFor(() => expect(h!.store.getState().copied).not.toBeNull());
    const paste = vi.spyOn(h!.store, "pasteSettings");
    await h!.user.click(search);
    await h!.user.keyboard("{Meta>}{Shift>}v{/Shift}{/Meta}");
    await h!.user.keyboard("{Meta>}{Alt>}{Shift>}v{/Shift}{/Alt}{/Meta}");
    expect(paste).not.toHaveBeenCalled();
  });

  it("have names a screen reader can read", async () => {
    await setup();
    expect(within(tools()).getByRole("group", { name: "Copy and paste settings" })).toBeInTheDocument();
    expect(copyButton()).toHaveAccessibleName("Copy settings");
    expect(pasteButton()).toHaveAccessibleName("Paste");
    expect(within(header()).getByRole("button", { name: "Add photos" })).toHaveAccessibleName("Add photos");
  });
});
