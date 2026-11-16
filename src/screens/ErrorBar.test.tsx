/** The bar at the bottom of the window: the memory notice (`docs/26` H-14) is a notice, below errors and other notices, and goes with the level. */
import { act, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

const SENTENCE = "Your Mac is short of memory, so Yellowmoon paused background work. Close other apps to speed it up again.";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

describe("the memory notice", () => {
  it("shows the core's sentence in the notice style, once, and goes when the core says the level is normal", async () => {
    h = renderApp();
    await screen.findByRole("banner");
    expect(screen.queryByText(SENTENCE)).toBeNull();

    act(() => h!.api.emitEvent("memory_notice", { message: SENTENCE }));
    const bar = (await screen.findByText(SENTENCE)).closest(".ym-error-bar") as HTMLElement;
    expect(bar).toHaveAttribute("role", "status");
    expect(bar).toHaveAttribute("data-kind", "notice");
    expect(within(bar).getByRole("button", { name: "Dismiss" })).toBeVisible();
    // It is a notice, not an error: no alert.
    expect(screen.queryByRole("alert")).toBeNull();

    // The same sentence again is still one bar.
    act(() => h!.api.emitEvent("memory_notice", { message: SENTENCE }));
    expect(screen.getAllByText(SENTENCE)).toHaveLength(1);

    act(() => h!.api.emitEvent("memory_notice", { message: null }));
    await waitFor(() => expect(screen.queryByText(SENTENCE)).toBeNull());
  });

  it("is hidden by Dismiss, and shows again when the core says it again", async () => {
    h = renderApp();
    await screen.findByRole("banner");
    act(() => h!.api.emitEvent("memory_notice", { message: SENTENCE }));
    await h.user.click(await screen.findByRole("button", { name: "Dismiss" }));
    expect(screen.queryByText(SENTENCE)).toBeNull();
    act(() => h!.api.emitEvent("memory_notice", { message: SENTENCE }));
    expect(await screen.findByText(SENTENCE)).toBeVisible();
  });

  it("waits behind an error, which comes first", async () => {
    h = renderApp();
    await screen.findByRole("banner");
    act(() => h!.api.emitEvent("memory_notice", { message: SENTENCE }));
    await screen.findByText(SENTENCE);
    act(() => h!.api.emitEvent("error", { scope: "app", message: "Something went wrong." }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong.");
    expect(screen.queryByText(SENTENCE)).toBeNull();
    await h.user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(await screen.findByText(SENTENCE)).toBeVisible();
  });
});
