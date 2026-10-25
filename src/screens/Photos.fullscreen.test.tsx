/**
 * Full screen on the Photos screen (`docs/11` §Full screen): F shows the photo alone, F or Esc goes back, Esc first ends a mode,
 * Crop and Pick white leave it first, the other keys keep working, and no hidden control keeps the focus. The window's own full screen
 * is the app's (`useFullScreen.test.tsx`); here the page runs on the mock, as in a browser, and only its layout changes.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockRect } from "../test/raf";
import { openPhotos, openTab, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

/** The mockup's state: photo 4 current, photos 3 to 5 picked. */
async function setup(): Promise<Harness> {
  h = renderApp({ current: 3, picked: [2, 3, 4] });
  await openPhotos(h);
  return h;
}

const photosScreen = (): HTMLElement => h!.container.querySelector<HTMLElement>('.ym-screen[data-screen="photos"]')!;
const photo = (): HTMLElement => screen.getByRole("main", { name: "Photo" });
// The hidden parts are out of the accessibility tree, names and all, so they are found by their place in the page.
const part = (selector: string): HTMLElement => photosScreen().querySelector<HTMLElement>(`:scope > ${selector}`)!;
const header = (): HTMLElement => part("header.ym-header");
const strip = (): HTMLElement => part("nav.ym-filmstrip");
const tools = (): HTMLElement => part('aside[aria-label="Tools"]');
const current = (): string | undefined => h!.store.currentPhoto()?.file_name;

/** The screen is in full screen: the photo alone, and the rest hidden from the eyes and from a screen reader. */
function expectFullScreen(): void {
  expect(photosScreen()).toHaveAttribute("data-fullscreen", "true");
  for (const el of [header(), strip(), tools()]) expect(el).not.toBeVisible();
  expect(screen.queryByRole("banner")).toBeNull();
  expect(screen.queryByRole("navigation", { name: "Photos" })).toBeNull();
  expect(screen.queryByRole("complementary", { name: "Tools" })).toBeNull();
  expect(photo()).toBeVisible();
}

/** The screen is back to its layout. */
function expectLayout(): void {
  expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
  for (const el of [header(), strip(), tools()]) expect(el).toBeVisible();
  expect(screen.getByRole("navigation", { name: "Photos" })).toBeInTheDocument();
}

describe("F", () => {
  it("shows the photo alone, says so once, and F again brings the header, the filmstrip and the panel back", async () => {
    await setup();
    expectLayout();
    await h!.user.keyboard("f");
    expectFullScreen();
    expect(screen.getByText("Full screen. Press F or Esc to leave.")).toHaveAttribute("aria-live", "polite");
    // The photo keeps its look chip.
    expect(within(photo()).getByText("Neutral · 100%")).toBeInTheDocument();
    await h!.user.keyboard("F");
    expectLayout();
    expect(screen.queryByText("Full screen. Press F or Esc to leave.")).toBeNull();
  });

  it("Esc leaves full screen", async () => {
    await setup();
    await h!.user.keyboard("f");
    expectFullScreen();
    await h!.user.keyboard("{Escape}");
    expectLayout();
  });

  it("Esc first ends Pick white, then leaves full screen", async () => {
    await setup();
    await h!.user.keyboard("w");
    await within(photo()).findByRole("application");
    await h!.user.keyboard("f");
    expectFullScreen();
    expect(within(photo()).getByRole("application")).toBeInTheDocument();
    await h!.user.keyboard("{Escape}");
    await waitFor(() => expect(within(photo()).queryByRole("application")).toBeNull());
    expectFullScreen();
    await h!.user.keyboard("{Escape}");
    expectLayout();
  });

  it("leaves the window as it is in a browser: nothing but the page changes", async () => {
    await setup();
    await h!.user.keyboard("f");
    expectFullScreen();
    await waitFor(() => expect(h!.api.viewport.rect).not.toBeNull());
    expect(document.fullscreenElement ?? null).toBeNull();
  });
});

describe("in full screen", () => {
  it("C leaves full screen and starts a crop", async () => {
    await setup();
    await h!.user.keyboard("f");
    expectFullScreen();
    await h!.user.keyboard("c");
    expect(await screen.findByRole("complementary", { name: "Crop" })).toBeVisible();
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
    expect(screen.getByRole("banner")).toBeVisible();
    // Done or Cancel brings back the panel, not full screen.
    await h!.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
  });

  it("W leaves full screen and turns Pick white on", async () => {
    await setup();
    await h!.user.keyboard("f");
    await h!.user.keyboard("w");
    expectLayout();
    expect(await screen.findByText("Click something neutral: grey or white.")).toBeInTheDocument();
    expect(within(photo()).getByRole("application")).toBeInTheDocument();
  });

  it("⌘E leaves full screen and opens Export; closing Export does not go back to full screen", async () => {
    await setup();
    await h!.user.keyboard("f");
    await h!.user.keyboard("{Meta>}e{/Meta}");
    expect(await screen.findByRole("complementary", { name: "Export" })).toBeVisible();
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
    await h!.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
  });

  it("Export opened another way (the File menu) ends full screen too", async () => {
    await setup();
    await h!.user.keyboard("f");
    expectFullScreen();
    h!.store.openExport();
    expect(await screen.findByRole("complementary", { name: "Export" })).toBeVisible();
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
    expect(screen.getByRole("banner")).toBeVisible();
  });

  it("⌘1–⌘7 leave full screen and show the tab", async () => {
    await setup();
    await h!.user.keyboard("f");
    await h!.user.keyboard("{Meta>}3{/Meta}");
    expectLayout();
    expect(screen.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
  });

  it("the arrows, P, backslash, Z and the number keys of the Look tab keep working", async () => {
    await setup();
    await h!.user.keyboard("f");
    await h!.user.keyboard("{ArrowRight}");
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
    await h!.user.keyboard("p");
    await waitFor(() => expect(h!.store.getState().session?.picked).toContain(4));
    await h!.user.keyboard("\\");
    expect(await within(photo()).findByText("Before · camera default")).toBeInTheDocument();
    await h!.user.keyboard("\\");
    await h!.user.keyboard("z");
    await waitFor(() => expect(h!.api.viewport.mode).toEqual({ kind: "percent", value: 100 }));
    const before = h!.store.currentSettings()?.values.look;
    await h!.user.keyboard("2");
    await waitFor(() => expect(h!.store.currentSettings()?.values.look).not.toBe(before));
    expectFullScreen();
  });

  it("a control that had the focus gives it up, so the arrows move between photos and not along the strip", async () => {
    await setup();
    const thumb = within(strip()).getByRole("button", { name: "Show DSC04206" });
    await h!.user.click(thumb);
    await waitFor(() => expect(current()).toBe("DSC04206.ARW"));
    expect(thumb).toHaveFocus();
    await h!.user.keyboard("f");
    expectFullScreen();
    expect(document.activeElement).toBe(document.body);
    await h!.user.keyboard("{ArrowLeft}");
    await waitFor(() => expect(current()).toBe("DSC04205.ARW"));
  });

  it("a tab that had the focus gives it up too", async () => {
    await setup();
    await openTab(h!, "Colour");
    expect(screen.getByRole("tab", { name: "Colour" })).toHaveFocus();
    await h!.user.keyboard("f");
    expect(document.activeElement).toBe(document.body);
  });

  it("Pick white counts the cell's own shape: the paper beside a 3:2 photo in a 16:10 window is not the photo", async () => {
    await setup();
    const input = vi.spyOn(h!.api, "viewportInput");
    await h!.user.keyboard("w");
    await h!.user.keyboard("f");
    const layer = await within(photo()).findByRole("application");
    // The cell is the whole window: a 3:2 photo leaves 45 px of paper on each side of 1440 × 900.
    mockRect(layer, { left: 0, top: 0, width: 1440, height: 900 });
    input.mockClear();
    fireEvent.click(layer, { clientX: 20, clientY: 450 });
    expect(input).not.toHaveBeenCalled();
    fireEvent.click(layer, { clientX: 720, clientY: 450 });
    await waitFor(() => expect(input).toHaveBeenCalledWith({ kind: "pick_white", x: 720, y: 450 }));
  });
});

describe("F does nothing", () => {
  it("in the Export panel, where Esc still goes back to the photos", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}e{/Meta}");
    await screen.findByRole("complementary", { name: "Export" });
    await h!.user.keyboard("f");
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
    await h!.user.keyboard("{Escape}");
    await screen.findByRole("complementary", { name: "Tools" });
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
  });

  it("while cropping", async () => {
    await setup();
    await h!.user.keyboard("c");
    await screen.findByRole("complementary", { name: "Crop" });
    await h!.user.keyboard("f");
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
    expect(screen.getByRole("complementary", { name: "Crop" })).toBeVisible();
  });

  it("in a text box, where it is a letter", async () => {
    await setup();
    const search = screen.getByRole("searchbox");
    await h!.user.click(search);
    await h!.user.keyboard("f");
    expect(search).toHaveValue("f");
    expect(photosScreen()).not.toHaveAttribute("data-fullscreen");
  });
});

describe("leaving the Photos screen", () => {
  it("ends full screen: ⌘W goes back to Start, and the next session opens in the usual layout", async () => {
    await setup();
    await h!.user.keyboard("f");
    expectFullScreen();
    await h!.user.keyboard("{Meta>}w{/Meta}");
    await screen.findByRole("button", { name: "Open files…" });
    expect(h!.container.querySelector("[data-fullscreen]")).toBeNull();
    await openPhotos(h!);
    expectLayout();
  });
});
