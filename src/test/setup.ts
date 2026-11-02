// Vitest setup: jest-dom matchers for Testing Library, and cleanup between tests.
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// glasscn's components stand on Radix, which measures with ResizeObserver and captures the pointer; jsdom has neither.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  };
}
if (typeof Element !== "undefined") {
  Element.prototype.hasPointerCapture ??= () => false;
  Element.prototype.setPointerCapture ??= () => undefined;
  Element.prototype.releasePointerCapture ??= () => undefined;
  Element.prototype.scrollIntoView ??= () => undefined;
}

// @xyflow/react (the node view's graph) reads the viewport's transform with a DOMMatrixReadOnly; jsdom has none. Only the scale and the
// translation of a 2D matrix are read, from the string `translate(x, y) scale(k)` or `matrix(k, 0, 0, k, x, y)` or `none`.
if (typeof globalThis.DOMMatrixReadOnly === "undefined") {
  globalThis.DOMMatrixReadOnly = class {
    m22 = 1;
    m41 = 0;
    m42 = 0;
    constructor(transform?: string) {
      const matrix = /matrix\(([^)]+)\)/.exec(transform ?? "");
      if (matrix) {
        const [, , , d, e, f] = matrix[1]!.split(",").map(Number);
        this.m22 = d ?? 1;
        this.m41 = e ?? 0;
        this.m42 = f ?? 0;
      }
    }
  } as unknown as typeof DOMMatrixReadOnly;
}

// jsdom has no canvas: its `getContext` prints "not implemented" for every call. The one canvas of the app (the colour warp's plane)
// is decoration and draws nothing without a context.
if (typeof HTMLCanvasElement !== "undefined") {
  HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
}

afterEach(() => {
  cleanup();
});
