/**
 * View › Nodes (`docs/11` §Menus): the menu item does what N and the Nodes button do. The shell sends `menu_action { action: "nodes" }`,
 * `runMenuAction` calls the Photos screen's `shortcutCommand`, and the screen toggles the node view with the same guards as the key.
 */
import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const nodesButton = (): HTMLElement => screen.getByRole("button", { name: /^Nodes$/ });

describe("View › Nodes", () => {
  it("opens and closes the node view like the Nodes button", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    expect(nodesButton()).toHaveAttribute("aria-pressed", "false");
    act(() => h!.api.emitEvent("menu_action", { action: "nodes" }));
    await waitFor(() => expect(nodesButton()).toHaveAttribute("aria-pressed", "true"));
    act(() => h!.api.emitEvent("menu_action", { action: "nodes" }));
    await waitFor(() => expect(nodesButton()).toHaveAttribute("aria-pressed", "false"));
  });

  it("does nothing in Crop, as N does not", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    await h.user.keyboard("c");
    await waitFor(() => expect(screen.getByRole("button", { name: /^Crop$/ })).toHaveAttribute("aria-pressed", "true"));
    act(() => h!.api.emitEvent("menu_action", { action: "nodes" }));
    expect(nodesButton()).toHaveAttribute("aria-pressed", "false");
  });

  it("does nothing while a dialog is open", async () => {
    h = renderApp({ current: 3 });
    await openPhotos(h);
    act(() => h!.store.openOverlay("preferences"));
    act(() => h!.api.emitEvent("menu_action", { action: "nodes" }));
    act(() => h!.store.closeOverlay());
    expect(nodesButton()).toHaveAttribute("aria-pressed", "false");
  });
});
