/**
 * Remove from filmstrip on the Photos screen (`docs/11` §Removing photos): the right-click menu of a thumbnail, Delete and Backspace
 * (not in a text box or on a slider), the notice with its Undo button, ⌘Z, and the Start screen when no photo is left.
 */
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import type { MockOptions } from "../api.mock";

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

const strip = (): HTMLElement => screen.getByRole("navigation", { name: "Photos" });
const count = (): number => h?.store.getState().session?.photos.length ?? 0;
const names = (): string[] => (h?.store.getState().session?.photos ?? []).map((p) => p.file_name.replace(/\.ARW$/, ""));
const notice = (): HTMLElement => screen.getByRole("status");

describe("the thumbnail's menu", () => {
  it("removes the photo that was right-clicked, says so, and leaves the one on screen alone", async () => {
    await setup();
    fireEvent.contextMenu(within(strip()).getByRole("button", { name: "Show DSC04207" }), { clientX: 50, clientY: 50 });
    const menu = await screen.findByRole("menu", { name: "DSC04207 actions" });
    await h!.user.click(within(menu).getByRole("menuitem", { name: "Remove from filmstrip" }));
    await waitFor(() => expect(count()).toBe(11));
    expect(names()).not.toContain("DSC04207");
    // DSC04204 is still on screen, with its place unchanged.
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04204.ARW");
    expect(within(strip()).getByText("4 of 11 · 3 picked")).toBeInTheDocument();
    expect(within(notice()).getByText("Removed DSC04207. The file is still on disk.")).toBeInTheDocument();
    expect(within(notice()).getByRole("button", { name: "Undo" })).toBeInTheDocument();
    expect(within(notice()).getByRole("button", { name: "Dismiss" })).toBeInTheDocument();
  });

  it("removes the picked photos and shows the next photo when the one on screen was among them", async () => {
    await setup();
    fireEvent.contextMenu(within(strip()).getByRole("button", { name: "Show DSC04201" }), { clientX: 50, clientY: 50 });
    const menu = await screen.findByRole("menu");
    const picked = within(menu).getByRole("menuitem", { name: /Remove picked photos/ });
    expect(picked).toHaveTextContent("3");
    await h!.user.click(picked);
    await waitFor(() => expect(count()).toBe(9));
    // Photos 3 to 5 went; 4 was on screen, so 6 is.
    expect(names()).not.toEqual(expect.arrayContaining(["DSC04203"]));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04206.ARW");
    expect(within(strip()).getByText("3 of 9 · 0 picked")).toBeInTheDocument();
    expect(within(notice()).getByText("Removed 3 photos. The files are still on disk.")).toBeInTheDocument();
  });

  it("offers Remove picked photos only while photos are picked", async () => {
    await setup({ current: 0, picked: [] });
    fireEvent.contextMenu(within(strip()).getByRole("button", { name: "Show DSC04203" }), { clientX: 50, clientY: 50 });
    const menu = await screen.findByRole("menu");
    expect(within(menu).getAllByRole("menuitem")).toHaveLength(1);
  });

  it("has no menu on the Export panel's strip", async () => {
    await setup();
    await h!.user.click(screen.getByRole("button", { name: /Export 3/ }));
    const photos = await screen.findByRole("navigation", { name: "Photos to export" });
    fireEvent.contextMenu(within(photos).getByRole("img", { name: /DSC04203/ }), { clientX: 5, clientY: 5 });
    expect(screen.queryByRole("menu")).toBeNull();
  });
});

describe("Delete and Backspace", () => {
  it("remove the photo on screen, and the next one is shown", async () => {
    await setup();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(11));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04205.ARW");
    await h!.user.keyboard("{Backspace}");
    await waitFor(() => expect(count()).toBe(10));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04206.ARW");
    expect(notice()).toHaveTextContent("Removed DSC04205. The file is still on disk.");
  });

  it("remove the last photo and show the previous one", async () => {
    await setup({ current: 11, picked: [] });
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(11));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04211.ARW");
  });

  it("do nothing in a text box", async () => {
    await setup();
    // The Look tab's search box.
    const search = screen.getByRole("searchbox");
    await h!.user.click(search);
    await h!.user.type(search, "abc");
    await h!.user.keyboard("{Backspace}{Delete}");
    expect(count()).toBe(12);
    expect(search).toHaveValue("ab");
  });

  it("do nothing on a slider's knob or in the value box next to it", async () => {
    await setup();
    await h!.user.click(screen.getByRole("tab", { name: "Light" }));
    const knob = await screen.findByRole("slider", { name: "Exposure" });
    knob.focus();
    await h!.user.keyboard("{Delete}{Backspace}");
    const box = screen.getByRole("textbox", { name: "Exposure value" });
    await h!.user.click(box);
    await h!.user.keyboard("{Backspace}{Delete}");
    expect(count()).toBe(12);
  });

  it("do nothing while cropping or picking white", async () => {
    await setup();
    await h!.user.keyboard("c");
    await screen.findByRole("complementary", { name: "Crop" });
    await h!.user.keyboard("{Delete}");
    expect(count()).toBe(12);
    await h!.user.keyboard("{Escape}");
    await h!.user.keyboard("w");
    await h!.user.keyboard("{Backspace}");
    expect(count()).toBe(12);
  });

  it("do nothing while the Export panel is open", async () => {
    await setup();
    await h!.user.keyboard("{Meta>}e{/Meta}");
    await screen.findByRole("navigation", { name: "Photos to export" });
    await h!.user.keyboard("{Delete}");
    expect(count()).toBe(12);
  });

  it("do nothing while a dialog is open", async () => {
    await setup();
    await h!.user.keyboard("{Meta>},{/Meta}");
    await screen.findByRole("dialog");
    await h!.user.keyboard("{Delete}");
    expect(count()).toBe(12);
  });
});

describe("Undo", () => {
  it("is a button of the notice and puts the photos back with their settings", async () => {
    await setup();
    await h!.store.update({ contrast: { kind: "manual", value: 30 } }, true);
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(11));
    await h!.user.click(within(notice()).getByRole("button", { name: "Undo" }));
    await waitFor(() => expect(count()).toBe(12));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04204.ARW");
    expect(within(strip()).getByText("4 of 12 · 3 picked")).toBeInTheDocument();
    expect(notice()).toHaveTextContent("Put back DSC04204.");
    expect(within(notice()).queryByRole("button", { name: "Undo" })).toBeNull();
    await waitFor(() => expect(h!.store.currentSettings()?.settings.contrast).toEqual({ kind: "manual", value: 30 }));
  });

  it("is ⌘Z while nothing changed since", async () => {
    await setup();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(11));
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(count()).toBe(12));
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04204.ARW");
  });

  it("goes away with Dismiss, but ⌘Z still works", async () => {
    await setup();
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(11));
    await h!.user.click(within(notice()).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("status")).toBeNull();
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await waitFor(() => expect(count()).toBe(12));
  });
});

describe("the last photo", () => {
  it("shows the Start screen with the notice, and Undo goes back to the photos", async () => {
    await setup({ photoCount: 2, current: 0, picked: [] });
    await h!.user.keyboard("{Delete}");
    await waitFor(() => expect(count()).toBe(1));
    await h!.user.keyboard("{Delete}");
    await screen.findByRole("heading", { name: "Drop Sony ARW files." });
    expect(screen.queryByRole("tab", { name: "Look" })).toBeNull();
    expect(notice()).toHaveTextContent("Removed DSC04202. The file is still on disk.");
    await h!.user.click(within(notice()).getByRole("button", { name: "Undo" }));
    await screen.findByRole("tab", { name: "Look" });
    expect(count()).toBe(1);
    expect(h!.store.currentPhoto()?.file_name).toBe("DSC04202.ARW");
  });

  it("takes ⌘Z on the Start screen too", async () => {
    await setup({ photoCount: 1, current: 0, picked: [] });
    await h!.user.keyboard("{Delete}");
    await screen.findByRole("heading", { name: "Drop Sony ARW files." });
    await h!.user.keyboard("{Meta>}z{/Meta}");
    await screen.findByRole("tab", { name: "Look" });
    expect(count()).toBe(1);
  });
});
