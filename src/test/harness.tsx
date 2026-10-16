/** A whole app on a fast mock, for the tests of the screens. */
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect } from "vitest";
import type { MockInstance } from "vitest";
import type { Api } from "../api";
import App from "../App";
import { createMockApi } from "../api.mock";
import type { MockApi, MockOptions } from "../api.mock";
import { createStore } from "../state/store";
import type { AppStore } from "../state/store";

/** A spy on `update_settings`. */
export type UpdateSpy = MockInstance<Api["updateSettings"]>;

export interface Harness {
  api: MockApi;
  store: AppStore;
  user: ReturnType<typeof userEvent.setup>;
  container: HTMLElement;
  unmount: () => void;
}

/** Renders the app against a mock that runs its timers at once. */
export function renderApp(options: MockOptions = {}): Harness {
  const api = createMockApi({ speed: Number.POSITIVE_INFINITY, ...options });
  const store = createStore(api, 0);
  const user = userEvent.setup();
  const view = render(<App store={store} />);
  return { api, store, user, container: view.container, unmount: view.unmount };
}

/** Opens the twelve photos from Start and goes to the Photos screen, with the current photo's settings loaded. */
export async function openPhotos(h: Harness): Promise<void> {
  await h.user.click(await screen.findByRole("button", { name: "Open files…" }));
  h.api.flush();
  await waitFor(() => expect(screen.getByRole("button", { name: /Open photos/ })).not.toHaveAttribute("aria-disabled"));
  await h.user.click(screen.getByRole("button", { name: /Open photos/ }));
  await screen.findByRole("tab", { name: "Look" });
  await waitFor(() => expect(h.store.currentSettings()).not.toBeNull());
  // The Look tab fills in when the settings arrive.
  await screen.findByRole("region", { name: "Automatic" });
}

/** The tool panel. */
export const tools = (): HTMLElement => screen.getByRole("complementary", { name: "Tools" });

/** Opens a tab by its label. */
export async function openTab(h: Harness, name: string): Promise<void> {
  await h.user.click(screen.getByRole("tab", { name }));
  await waitFor(() => expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true"));
}

/** The text of a slider's value box. */
export const valueOf = (label: string): string => (screen.getByRole("textbox", { name: `${label} value` }) as HTMLInputElement).value;

/** The slider (glasscn's Radix knob, `role="slider"`) of a labelled row. */
export const sliderOf = (label: string): HTMLElement => screen.getByRole("slider", { name: label });

/** The name a slider has from its row's label (`aria-labelledby`), as a person reads it. */
export const sliderName = (slider: HTMLElement): string | undefined =>
  slider.getAttribute("aria-labelledby")?.split(/\s+/).map((id) => document.getElementById(id)?.textContent ?? "").join(" ");

export { screen, waitFor, within };
