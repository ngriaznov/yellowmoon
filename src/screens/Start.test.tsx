import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const dropOnWindow = (...names: string[]): void => {
  const data = { types: ["Files"], files: names.map((name) => ({ name })), dropEffect: "none" };
  act(() => {
    fireEvent.dragEnter(window, { dataTransfer: data });
    fireEvent.drop(window, { dataTransfer: data });
  });
};

describe("Start, nothing open", () => {
  it("shows the header, the yellow block, the drop zone and the steps of the mockup", () => {
    h = renderApp({ looksImported: false });
    expect(screen.getByText("Yellowmoon")).toBeInTheDocument();
    expect(screen.getByText("Drop ARW files to start")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open files ⌘O" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open folder ⇧⌘O" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
    expect(screen.getByText(/Every photo is developed automatically/)).toBeInTheDocument();
    expect(screen.getByText("Any Sony camera · every kind of ARW")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Drop files or a folder" })).toBeInTheDocument();
    expect(screen.getByText("ARW only, as many as you like")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open files…" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open folder…" })).toBeInTheDocument();
    const steps = within(screen.getByRole("region", { name: "How it works" }));
    expect(steps.getByText("Adjust, if you want")).toBeInTheDocument();
    expect(steps.getByText("JPEG, TIFF, PNG or DNG. Done.")).toBeInTheDocument();
  });

  it("has Export disabled, with the reason", () => {
    h = renderApp();
    const exportCell = screen.getByRole("button", { name: /Export/ });
    expect(exportCell).toHaveAttribute("aria-disabled", "true");
    expect(exportCell).toHaveAccessibleDescription("Open photos to export");
  });

  it("marks the first step as the current one", () => {
    h = renderApp();
    const steps = screen.getByRole("region", { name: "How it works" });
    const on = steps.querySelectorAll("[data-on]");
    expect(on).toHaveLength(1);
    expect(on[0]).toHaveTextContent("Open");
  });

  it("says the ARRI looks are not imported yet, and offers the wizard", async () => {
    h = renderApp({ looksImported: false });
    expect(await screen.findByText("Not imported yet")).toBeInTheDocument();
    await h.user.click(screen.getByRole("button", { name: "Import →" }));
    expect(screen.getByRole("heading", { name: "Choose what to import" })).toBeInTheDocument();
  });

  it("counts the imported ARRI looks", async () => {
    h = renderApp();
    expect(await screen.findByText("7 looks imported")).toBeInTheDocument();
    expect(screen.queryByText("Not imported yet")).toBeNull();
  });

  it("shows the default look and changes it from a menu", async () => {
    h = renderApp();
    const row = (await screen.findByText("Default look")).closest(".ym-about-row") as HTMLElement;
    expect(within(row).getByText("Neutral")).toBeInTheDocument();
    await h.user.click(within(row).getByRole("button", { name: "Change" }));
    const menu = screen.getByRole("menu", { name: "Default look" });
    expect(within(menu).getAllByRole("menuitemradio")).toHaveLength(9);
    expect(within(menu).getByRole("menuitemradio", { name: "Neutral" })).toHaveAttribute("aria-checked", "true");
    await h.user.click(within(menu).getByRole("menuitemradio", { name: "Signature" }));
    await waitFor(() => expect(within(row).getByText("Signature")).toBeInTheDocument());
    expect(screen.queryByRole("menu")).toBeNull();
    expect((await h.api.preferencesGet()).default_look).toBe("builtin:signature");
  });

  it("names a default look by its library when two libraries have a look of that name", async () => {
    h = renderApp({ allLibraries: true });
    const row = (await screen.findByText("Default look")).closest(".ym-about-row") as HTMLElement;
    await h.user.click(within(row).getByRole("button", { name: "Change" }));
    const menu = screen.getByRole("menu", { name: "Default look" });
    const names = within(menu).getAllByRole("menuitemradio").map((item) => item.textContent);
    expect(names).toHaveLength(29);
    expect(names).toContain("Neutral");
    expect(names).toContain("Monochrome");
    expect(names).toContain("Film warm · LogC4");
    expect(names).toContain("Film warm · LogC3");
    expect(names).toContain("Film warm · LogC3 to Rec.709");
    expect(names).toContain("Film cool · Your LUTs");
    // An ALF2 title already says its strength, and the same title is in the LogC3 to Rec.709 library too.
    expect(names).toContain("Film warm · Low");
    // Picking one shows the same words in the row.
    await h.user.click(within(menu).getByRole("menuitemradio", { name: "Film warm · LogC3" }));
    await waitFor(() => expect(within(row).getByText("Film warm · LogC3")).toBeInTheDocument());
    expect((await h.api.preferencesGet()).default_look).toBe("arri:3000:film-warm-logc3");
  });

  it("Open files and Open folder use the system dialog", async () => {
    h = renderApp();
    const spy = vi.spyOn(h.api, "openDialog");
    await h.user.click(screen.getByRole("button", { name: "Open files…" }));
    expect(spy).toHaveBeenLastCalledWith("files");
    await waitFor(() => expect(screen.getByRole("heading", { name: /photos/i, level: 2 })).toBeInTheDocument());
  });

  it("⌘O and ⇧⌘O open files and a folder", async () => {
    h = renderApp();
    const spy = vi.spyOn(h.api, "openDialog");
    await h.user.keyboard("{Meta>}o{/Meta}");
    expect(spy).toHaveBeenLastCalledWith("files");
    h.unmount();
    h = renderApp();
    const second = vi.spyOn(h.api, "openDialog");
    await h.user.keyboard("{Meta>}{Shift>}o{/Shift}{/Meta}");
    expect(second).toHaveBeenLastCalledWith("folder");
  });

  it("opens files dropped anywhere in the window", async () => {
    h = renderApp();
    const spy = vi.spyOn(h.api, "openPaths");
    dropOnWindow("DSC04201.ARW", "notes.txt");
    await waitFor(() => expect(spy).toHaveBeenCalledWith(["DSC04201.ARW"]));
    await screen.findByRole("heading", { level: 2, name: /Developing 12 photos|12 photos ready/ });
  });

  it("highlights the drop zone while files are dragged over the window", () => {
    h = renderApp();
    act(() => {
      fireEvent.dragEnter(window, { dataTransfer: { types: ["Files"], files: [] } });
    });
    expect(screen.getByRole("group", { name: "Drop files or a folder" })).toHaveAttribute("data-active", "true");
  });

  it("says so when the drop holds no ARW files, and opens nothing", async () => {
    h = renderApp();
    const spy = vi.spyOn(h.api, "openPaths");
    dropOnWindow("holiday.jpg", "notes.txt");
    expect(await screen.findByRole("alert")).toHaveTextContent("No ARW files in what you dropped.");
    expect(spy).not.toHaveBeenCalled();
    expect(screen.getByRole("group", { name: "Drop files or a folder" })).toBeInTheDocument();
  });
});

describe("Start, developing", () => {
  it("shows Developing N photos with a progress bar and a tagged list", async () => {
    h = renderApp({ holdAt: 5.4 });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    const title = await screen.findByRole("heading", { level: 2, name: "Developing 12 photos" });
    expect(title).toBeInTheDocument();
    // The machine may be busy: give the mock's timers room.
    await waitFor(() => expect(within(screen.getByRole("list", { name: "Photos being developed" })).getAllByText("Ready")).toHaveLength(5), { timeout: 8000 });
    const list = within(screen.getByRole("list", { name: "Photos being developed" }));
    expect(list.getAllByRole("listitem")).toHaveLength(12);
    expect(list.getAllByText("Ready")).toHaveLength(5);
    expect(list.getAllByText("Developing")).toHaveLength(1);
    expect(list.getAllByText("Waiting")).toHaveLength(6);
    expect(list.getAllByText("Auto exposure, white balance and noise")).toHaveLength(5);
    expect(list.getByText("Decoding and demosaicing")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Developing photos" })).toHaveAttribute("aria-valuenow", "45");
    expect(screen.getByText("You can start looking while the rest finishes.")).toBeInTheDocument();
  });

  it("shows the folder name and date in the header, and step 2 as current", async () => {
    h = renderApp({ holdAt: 2 });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    expect(await screen.findByText("Lisbon")).toBeInTheDocument();
    expect(screen.getByText("2026-10-04 · 12 photos")).toBeInTheDocument();
    const on = screen.getByRole("region", { name: "How it works" }).querySelectorAll("[data-on]");
    expect(on[0]).toHaveTextContent("Adjust, if you want");
  });

  it("enables Open photos as soon as the first photo is ready", async () => {
    h = renderApp({ holdAt: 0.2 });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    const open = await screen.findByRole("button", { name: /Open photos/ });
    expect(open).toHaveAttribute("aria-disabled", "true");
    h.api.configure({ holdAt: 3 });
    h.api.flush();
    await waitFor(() => expect(screen.getByRole("button", { name: /Open photos/ })).not.toHaveAttribute("aria-disabled"));
  });

  it("says everything is ready when all photos are developed", async () => {
    h = renderApp();
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    expect(await screen.findByRole("heading", { level: 2, name: "12 photos ready" })).toBeInTheDocument();
    expect(screen.getByText("Everything is developed and ready to export.")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Developing photos" })).toHaveAttribute("aria-valuenow", "100");
  });

  it("lists a file that can't be developed as Skipped, with its reason, and says how many", async () => {
    h = renderApp({ skipped: { DSC04207: "This file is from a camera mode we can't read yet." } });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    expect(await screen.findByRole("heading", { level: 2, name: "11 photos ready" })).toBeInTheDocument();
    const list = within(screen.getByRole("list", { name: "Photos being developed" }));
    expect(list.getAllByText("Skipped")).toHaveLength(1);
    expect(list.getByText("This file is from a camera mode we can't read yet.")).toBeInTheDocument();
    expect(screen.getByText("1 file skipped. Everything else is developed and ready to export.")).toBeInTheDocument();
  });

  it("uses the plural for several skipped files", async () => {
    h = renderApp({ skipped: { DSC04202: "Can't read.", DSC04203: "Can't read." } });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    expect(await screen.findByText("2 files skipped. Everything else is developed and ready to export.")).toBeInTheDocument();
  });

  it("goes to Photos with Open photos, on a photo that is ready", async () => {
    h = renderApp({ skipped: { DSC04201: "Can't read this one." } });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    await waitFor(() => expect(screen.getByRole("button", { name: /Open photos/ })).not.toHaveAttribute("aria-disabled"));
    await h.user.click(screen.getByRole("button", { name: /Open photos/ }));
    expect(await screen.findByRole("tab", { name: "Look" })).toBeInTheDocument();
    expect(h.store.currentPhoto()?.file_name).toBe("DSC04202.ARW");
  });

  it("Start over closes the session and shows the drop zone again", async () => {
    h = renderApp();
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    await h.user.click(await screen.findByRole("button", { name: "Start over" }));
    expect(await screen.findByRole("group", { name: "Drop files or a folder" })).toBeInTheDocument();
    expect((await h.api.sessionState()).photos).toHaveLength(0);
  });

  it("says when nothing could be developed", async () => {
    const skipped = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`DSC042${String(i + 1).padStart(2, "0")}`, "Can't read."]));
    h = renderApp({ skipped });
    await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
    h.api.flush();
    expect(await screen.findByRole("heading", { level: 2, name: "No photos ready" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Open photos/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("shows an error from the core in the error bar and can dismiss it", async () => {
    h = renderApp();
    h.api.failNext("openDialog", { scope: "open", message: "That folder can't be read. Choose another." });
    await h.user.click(await screen.findByRole("button", { name: "Open folder…" }));
    const bar = await screen.findByRole("alert");
    expect(bar).toHaveTextContent("That folder can't be read. Choose another.");
    await h.user.click(within(bar).getByRole("button", { name: "Dismiss" }));
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
