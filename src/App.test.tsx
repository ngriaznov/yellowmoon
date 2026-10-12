import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SCREENS } from "./App";
import { openPhotos, renderApp } from "./test/harness";
import type { Harness } from "./test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

const current = (container: HTMLElement): string | null => container.querySelector("[data-screen]")?.getAttribute("data-screen") ?? null;

describe("App", () => {
  it("opens on the Start screen", () => {
    h = renderApp();
    expect(current(h.container)).toBe("start");
    expect(screen.getByRole("heading", { level: 1, name: "Drop Sony ARW files." })).toBeInTheDocument();
    expect(screen.getByText("Drop ARW files to start")).toBeInTheDocument();
  });

  it("shows the moon mark next to the name, hidden from assistive technology", () => {
    h = renderApp();
    const moon = h.container.querySelector(".ym-brand-title .ym-moon");
    expect(moon).not.toBeNull();
    expect(moon).toHaveAttribute("aria-hidden", "true");
    expect(h.container.querySelectorAll(".ym-moon")).toHaveLength(1);
  });

  it("routes Start, Photos and the wizard", async () => {
    expect(SCREENS).toEqual(["start", "photos", "import-looks"]);
    h = renderApp();
    await openPhotos(h);
    expect(current(h.container)).toBe("photos");
    await h.user.click(screen.getByRole("button", { name: "Import ARRI looks →" }));
    expect(current(h.container)).toBe("import-looks");
    expect(screen.getByRole("heading", { level: 1, name: "Choose what to import" })).toBeInTheDocument();
    await h.user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(current(h?.container as HTMLElement)).toBe("photos"));
  });

  it("goes back to Start when the session closes", async () => {
    h = renderApp();
    await openPhotos(h);
    await act(async () => {
      await h?.store.closeSession();
    });
    expect(current(h.container)).toBe("start");
  });

  it("shows an error from the core on any screen and lets it be dismissed", async () => {
    h = renderApp();
    await openPhotos(h);
    h.api.failNext("looksList", { scope: "looks", message: "The looks folder can't be read. Check its permissions." });
    await act(async () => {
      await h?.store.refreshLooks();
    });
    expect(await screen.findByRole("alert")).toHaveTextContent("The looks folder can't be read. Check its permissions.");
  });
});
