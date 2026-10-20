import { afterEach, describe, expect, it, vi } from "vitest";
import { createMockApi } from "../api.mock";
import { checkNotice, runMenuAction } from "./menu";
import { createStore } from "./store";

/** A manual contrast or saturation (`docs/19` section 5: Auto or a value). */
const manual = (value: number) => ({ kind: "manual", value }) as const;

describe("checkNotice (Check for Updates…)", () => {
  it("says an update that was just found downloads when the cell is pressed", () => {
    expect(checkNotice({ kind: "available", version: "0.2.0" }, { phase: "available", version: "0.2.0" })).toBe(
      "Version 0.2.0 is available. It downloads when you press Update available.",
    );
  });

  it("does not point at 'Update available' while the update downloads", () => {
    expect(checkNotice({ kind: "available", version: "0.2.0" }, { phase: "downloading", version: "0.2.0", percent: 30 })).toBe("Version 0.2.0 is downloading.");
    expect(checkNotice({ kind: "available", version: "0.2.0" }, { phase: "downloading", version: "0.2.0" })).toBe("Version 0.2.0 is downloading.");
  });

  it("does not point at 'Update available' once the update is ready", () => {
    expect(checkNotice({ kind: "available", version: "0.2.0" }, { phase: "ready", version: "0.2.0" })).toBe("Version 0.2.0 is ready. Press Update ready to restart.");
  });

  it("says when there is nothing new, and when the check failed", () => {
    expect(checkNotice({ kind: "up_to_date" }, { phase: "idle" })).toBe("Yellowmoon is up to date.");
    expect(checkNotice({ kind: "error", message: "offline" }, { phase: "error", message: "offline" })).toBe(
      "Yellowmoon couldn't check for updates. Check your internet connection and try again.",
    );
  });
});

describe("the Edit menu's settings items", () => {
  const stops: Array<() => void> = [];
  afterEach(() => {
    for (const stop of stops.splice(0)) stop();
  });

  it("copy, paste and paste to picked do what the buttons and the keys do", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    stops.push(store.connect());
    await store.load();
    await store.openDialog("files");
    api.flush();
    await vi.waitFor(() => expect(store.getState().session?.photos.every((p) => p.status.kind === "ready")).toBe(true));
    await store.openPhotos();
    await vi.waitFor(() => expect(store.currentSettings()).not.toBeNull());
    await store.update({ contrast: manual(30) });
    const copy = vi.spyOn(store, "copySettings");
    const paste = vi.spyOn(store, "pasteSettings");

    await runMenuAction(store, "copy_settings");
    expect(copy).toHaveBeenCalledTimes(1);
    expect(store.getState().copied?.source).toBe("DSC04201");
    await runMenuAction(store, "paste_settings");
    expect(paste).toHaveBeenLastCalledWith("this");
    await runMenuAction(store, "paste_settings_to_picked");
    expect(paste).toHaveBeenLastCalledWith("picked");
    // Nobody is picked.
    expect(store.getState().notice).toBe("No photos are picked.");
  });

  it("arrive from the shell as menu_action events", async () => {
    const api = createMockApi({ speed: Number.POSITIVE_INFINITY });
    const store = createStore(api, 0);
    const handler = vi.fn();
    store.menuHandler = handler;
    stops.push(store.connect());
    for (const action of ["copy_settings", "paste_settings", "paste_settings_to_picked"] as const) {
      api.emitEvent("menu_action", { action });
      expect(handler).toHaveBeenLastCalledWith(action);
    }
  });
});
