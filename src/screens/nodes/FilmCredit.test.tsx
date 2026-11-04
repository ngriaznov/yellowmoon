import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FILM_CREDIT } from "../../fixtures/filmCredit";
import type { AppStore } from "../../state/store";
import { FilmCredit } from "./FilmCredit";

function fakeStore(): { store: AppStore; open: ReturnType<typeof vi.fn>; fail: ReturnType<typeof vi.fn> } {
  const open = vi.fn().mockResolvedValue(undefined);
  const fail = vi.fn();
  return { store: { api: { openUrl: open }, reportError: fail } as unknown as AppStore, open, fail };
}

describe("the credit line", () => {
  it("reads as the line of the film data, with the author's name as the only link on the Film tab", async () => {
    const { store, open } = fakeStore();
    const { container } = render(<FilmCredit store={store} />);
    expect(container.textContent).toBe(FILM_CREDIT);
    expect(screen.getAllByRole("button")).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: "Andrea Volpato (spektrafilm)" }));
    expect(open).toHaveBeenCalledWith("https://github.com/andreavolpato/spektrafilm");
  });

  it("links the licence too in About, and opens each page in the system browser", async () => {
    const { store, open } = fakeStore();
    const { container } = render(<FilmCredit store={store} licenceLink />);
    expect(container.textContent).toBe(FILM_CREDIT);
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["Andrea Volpato (spektrafilm)", "CC BY-SA 4.0"]);
    await userEvent.click(screen.getByRole("button", { name: "CC BY-SA 4.0" }));
    expect(open).toHaveBeenCalledWith("https://creativecommons.org/licenses/by-sa/4.0/");
  });

  it("says what went wrong when the page cannot be opened", async () => {
    const { store, open, fail } = fakeStore();
    const error = { scope: "system", message: "The page could not be opened." };
    open.mockRejectedValue(error);
    render(<FilmCredit store={store} />);
    await userEvent.click(screen.getByRole("button", { name: /Andrea Volpato/ }));
    await vi.waitFor(() => expect(fail).toHaveBeenCalledWith(error));
  });

  it("draws the line the core gave when it has one", () => {
    const { store } = fakeStore();
    const { container } = render(<FilmCredit store={store} credit="Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with anyone." />);
    expect(container.textContent).toMatch(/Not affiliated with anyone\.$/);
  });
});
