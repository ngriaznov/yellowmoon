import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useAnnouncer } from "./useAnnouncer";

describe("the live region's text", () => {
  it("changes every time, even for the same sentence, so a screen reader says it again", () => {
    const { result } = renderHook(() => useAnnouncer());
    const seen: string[] = [];
    for (let n = 0; n < 3; n += 1) {
      act(() => result.current.announcer.say("Before."));
      seen.push(result.current.text);
    }
    expect(new Set(seen.slice(0, 2)).size).toBe(2);
    expect(seen.every((t) => t.trim() === "Before.")).toBe(true);
    expect(seen[2]).toBe(seen[0]);
  });
});
