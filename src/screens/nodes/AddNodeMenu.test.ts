import { describe, expect, it } from "vitest";
import { blockedText } from "./AddNodeMenu";

describe("which Add node items are off (docs/25 §5.1)", () => {
  it("lets each item add what fits in 64 nodes and says why when it does not", () => {
    // Corrector and Film add one node.
    expect(blockedText(1, 63)).toBeNull();
    expect(blockedText(1, 64)).toBe("A photo can have up to 64 nodes.");
    // Parallel and Layer add two: off from 63.
    expect(blockedText(2, 62)).toBeNull();
    expect(blockedText(2, 63)).toBe("This adds two nodes, and a photo can have up to 64.");
    expect(blockedText(2, 64)).toBe("A photo can have up to 64 nodes.");
    // Splitter and combiner adds five: off from 60.
    expect(blockedText(5, 59)).toBeNull();
    expect(blockedText(5, 60)).toBe("This adds five nodes, and a photo can have up to 64.");
    expect(blockedText(5, 64)).toBe("A photo can have up to 64 nodes.");
  });
});
