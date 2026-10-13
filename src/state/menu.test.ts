import { describe, expect, it } from "vitest";
import { checkNotice } from "./menu";

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
