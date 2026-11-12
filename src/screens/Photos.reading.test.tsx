import { screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { openPhotos, renderApp } from "../test/harness";
import type { Harness } from "../test/harness";

let h: Harness | undefined;
afterEach(() => {
  h?.unmount();
  h = undefined;
});

describe("Photos, adding many files (docs/26 H-10)", () => {
  it("says how far the reading has got in the header, with Cancel, and Cancel says how many files were opened", async () => {
    h = renderApp();
    await openPhotos(h);
    // The next open reads 2000 files, slowly: the mock plays the core.
    h.api.configure({ slowOpen: 2000, speed: 1 });
    await h.user.click(screen.getByRole("button", { name: "Add photos" }));
    expect(await screen.findByText(/^Reading \d+ of 2000 files…$/, undefined, { timeout: 5000 })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add photos" })).toBeNull();
    await h.user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(await screen.findByText(/^Stopped\. .* opened\.$/)).toBeInTheDocument();
    expect(screen.queryByText(/Reading \d+ of/)).toBeNull();
    expect(await screen.findByRole("button", { name: "Add photos" })).toBeInTheDocument();
  });
});
