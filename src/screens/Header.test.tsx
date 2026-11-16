/** The header's two jobs that only the real shell makes matter: dragging the window, and holding an update back while the core exports. */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";
import { updater } from "../updater";

let h: Harness | undefined;
afterEach(() => {
  act(() => updater.applyExternal({ phase: "idle" }));
  vi.restoreAllMocks();
  h?.unmount();
  h = undefined;
});

describe("dragging the window", () => {
  it("is possible by the folder cell and the gaps of the header, on every screen that has one (titleBarStyle Overlay has no title bar)", async () => {
    h = renderApp();
    // Start.
    let header = (await screen.findByRole("banner")) as HTMLElement;
    expect(header).toHaveAttribute("data-tauri-drag-region", "");
    expect(header.querySelector(".ym-brand")).toHaveAttribute("data-tauri-drag-region", "deep");
    // Photos.
    await openPhotos(h);
    header = screen.getByRole("banner");
    expect(header).toHaveAttribute("data-tauri-drag-region", "");
    expect(header.querySelector(".ym-brand")).toHaveAttribute("data-tauri-drag-region", "deep");
    // A button is never a drag handle.
    expect(header.querySelectorAll("button[data-tauri-drag-region]")).toHaveLength(0);
    // The wizard.
    h.store.go("import-looks");
    await waitFor(() => expect(document.querySelector('[data-screen="import-looks"] .ym-brand')).toHaveAttribute("data-tauri-drag-region", "deep"));
    expect(document.querySelector('[data-screen="import-looks"] > header')).toHaveAttribute("data-tauri-drag-region", "");
  });
});

describe("Update ready while the core exports", () => {
  it("waits for an export the core says is busy, also one this window did not start", async () => {
    h = renderApp();
    await openPhotos(h);
    act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
    const pill = await screen.findByRole("button", { name: /Update ready/ });
    expect(pill).not.toHaveAttribute("aria-disabled");
    act(() => h!.api.emitEvent("export_busy", { busy: true }));
    await waitFor(() => expect(screen.getByRole("button", { name: /Update ready/ })).toHaveAttribute("aria-disabled", "true"));
    expect(screen.getByRole("button", { name: /Update ready/ })).toHaveAccessibleDescription("Restarts after the export finishes");
    act(() => h!.api.emitEvent("export_busy", { busy: false }));
    await waitFor(() => expect(screen.getByRole("button", { name: /Update ready/ })).not.toHaveAttribute("aria-disabled"));
  });

  it("restarts when nothing is exporting", async () => {
    h = renderApp();
    await openPhotos(h);
    const restart = vi.spyOn(updater, "restart").mockResolvedValue(true);
    act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
    await h.user.click(await screen.findByRole("button", { name: /Update ready/ }));
    await waitFor(() => expect(restart).toHaveBeenCalledWith(false));
  });

  it("asks the core at the click, because the restart goes past the shell's question: an export that began unheard holds it back", async () => {
    h = renderApp();
    await openPhotos(h);
    const restart = vi.spyOn(updater, "restart").mockResolvedValue(true);
    act(() => updater.applyExternal({ phase: "ready", version: "0.1.1" }));
    const pill = await screen.findByRole("button", { name: /Update ready/ });
    // The queue went busy without this window hearing about it.
    vi.spyOn(h.api, "exportBusy").mockResolvedValue(true);
    await h.user.click(pill);
    await waitFor(() => expect(h!.store.getState().exportBusy).toBe(true));
    expect(restart).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /Update ready/ })).toHaveAttribute("aria-disabled", "true");
  });

  it("other updater states are not asked about", async () => {
    h = renderApp();
    await openPhotos(h);
    const download = vi.spyOn(updater, "download").mockResolvedValue();
    const busy = vi.spyOn(h.api, "exportBusy");
    act(() => updater.applyExternal({ phase: "available", version: "0.1.1" }));
    await h.user.click(await screen.findByRole("button", { name: /Update available/ }));
    expect(download).toHaveBeenCalled();
    expect(busy).not.toHaveBeenCalled();
  });
});

describe("the folder cell of an album (docs/31 section 6)", () => {
  it("shows the album's name in place of the folder's, and EDITED in place of the date while it has changes", async () => {
    h = renderApp({ albums: ["Lisbon"] });
    await h.store.openPaths(["~/Pictures/Lisbon/Lisbon.ymalbum"]);
    h.api.flush();
    await h.store.openPhotos();
    await screen.findByRole("tab", { name: "Look" });
    const header = screen.getByRole("banner");
    expect(header.querySelector(".ym-brand-name")).toHaveTextContent("Lisbon");
    expect(header.querySelector(".ym-brand-sub")).toHaveTextContent("2026-10-04 · 12 photos");
    await h.store.togglePick();
    await waitFor(() => expect(header.querySelector(".ym-brand-sub")).toHaveTextContent("12 photos · edited"));
    expect(header.querySelector(".ym-brand-sub")).not.toHaveTextContent("2026");
    // Save brings the date back.
    await h.store.saveAlbum(false);
    await waitFor(() => expect(header.querySelector(".ym-brand-sub")).toHaveTextContent("2026-10-04 · 12 photos"));
  });

  it("asks 'Save changes to the album Lisbon?' with Save as the default when Close Session is chosen", async () => {
    h = renderApp({ albums: ["Lisbon"] });
    await h.store.openPaths(["~/Pictures/Lisbon/Lisbon.ymalbum"]);
    h.api.flush();
    await h.store.openPhotos();
    await screen.findByRole("tab", { name: "Look" });
    await h.store.togglePick();
    await waitFor(() => expect(h!.store.getState().session?.album?.edited).toBe(true));
    h.api.emitEvent("menu_action", { action: "close_session" });
    const dialog = await screen.findByRole("dialog", { name: "Save changes to the album Lisbon?" });
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "Save" })).toHaveFocus());
    expect(within(dialog).getByRole("button", { name: "Don't Save" })).toBeInTheDocument();
    await h.user.click(within(dialog).getByRole("button", { name: "Cancel" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(h.store.getState().session).not.toBeNull();
    h.api.emitEvent("menu_action", { action: "close_session" });
    await h.user.click(await screen.findByRole("button", { name: "Don't Save" }));
    await waitFor(() => expect(h!.store.getState().session).toBeNull());
  });
});
