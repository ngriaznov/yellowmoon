import { describe, expect, it } from "vitest";
import { ViewportInputQueue } from "./viewportInputQueue";
import type { StreamInput } from "./viewportInputQueue";
import { fitZoomExact, fitZoomOf, fitZoomPercent, MAX_ZOOM_PERCENT, PAN_STEP, PAN_STEP_SHIFT, panKeyDelta, showsWhole, zoomAfter, zoomKey, zoomStep } from "./zoom";

const photo = { width_px: 8640, height_px: 5760 };
const cell = { width: 1020, height: 680, scale: 2 };

describe("fitZoomPercent", () => {
  it("is the cell's device width over the photo's width for a landscape photo in the 3:2 cell: Fit 24%", () => {
    expect(fitZoomPercent(cell, photo, null)).toBe(24);
    expect(fitZoomPercent(cell, photo, { orientation: 0, crop: null })).toBe(24);
  });

  it("follows the height for a portrait photo, after a quarter turn", () => {
    // 5760 × 8640 after the turn: the cell's 1360 device px of height hold 8640 px.
    expect(fitZoomPercent(cell, photo, { orientation: 90, crop: null })).toBe(16);
    expect(fitZoomPercent(cell, photo, { orientation: 270, crop: null })).toBe(16);
    expect(fitZoomPercent(cell, photo, { orientation: 180, crop: null })).toBe(24);
  });

  it("follows the crop: a smaller picture fits at a larger zoom", () => {
    const crop = { rect: { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }, angle_deg: 0, aspect: "free" as const };
    expect(fitZoomPercent(cell, photo, { orientation: 0, crop })).toBe(47);
  });

  it("has no answer before the cell or the photo has a size", () => {
    expect(fitZoomPercent({ width: 0, height: 0, scale: 2 }, photo, null)).toBeNull();
    expect(fitZoomPercent(cell, { width_px: 0, height_px: 0 }, null)).toBeNull();
  });
});

describe("zoomAfter", () => {
  it("zooms in from Fit by the factor, from the Fit zoom", () => {
    expect(zoomAfter({ kind: "fit" }, 2, 24)).toEqual({ kind: "percent", value: 48 });
    // The core snaps a zoom within 0.01 % of the Fit zoom to Fit (`zoom_about`), so a hair above 1 is still Fit.
    expect(zoomAfter({ kind: "fit" }, 1.001, 24).kind).toBe("percent");
    expect(zoomAfter({ kind: "fit" }, 1.0001, 24)).toEqual({ kind: "fit" });
  });

  it("starts from 25 % when the Fit zoom is not known", () => {
    expect(zoomAfter({ kind: "fit" }, 2, null)).toEqual({ kind: "percent", value: 50 });
  });

  it("stays at Fit when zooming out at Fit", () => {
    expect(zoomAfter({ kind: "fit" }, 0.5, 24)).toEqual({ kind: "fit" });
    expect(zoomAfter({ kind: "fit" }, 1, 24)).toEqual({ kind: "fit" });
  });

  it("multiplies a fixed zoom, and goes back to Fit at or below the Fit zoom", () => {
    expect(zoomAfter({ kind: "percent", value: 100 }, 1.5, 24)).toEqual({ kind: "percent", value: 150 });
    expect(zoomAfter({ kind: "percent", value: 100 }, 0.2, 24)).toEqual({ kind: "fit" });
    expect(zoomAfter({ kind: "percent", value: 100 }, 0.24, 24)).toEqual({ kind: "fit" });
    expect(zoomAfter({ kind: "percent", value: 100 }, 0.5, null)).toEqual({ kind: "percent", value: 50 });
  });

  it("stops at 800 %", () => {
    expect(zoomAfter({ kind: "percent", value: 700 }, 3, 24)).toEqual({ kind: "percent", value: MAX_ZOOM_PERCENT });
  });

  it("keeps two decimals, so a run of small steps does not stall", () => {
    let mode = zoomAfter({ kind: "fit" }, 1.04, 24);
    for (let i = 0; i < 10; i++) mode = zoomAfter(mode, 1.01, 24);
    expect(mode.kind === "percent" && mode.value).toBeGreaterThan(27);
  });

  it("ignores a factor that is not a positive number", () => {
    expect(zoomAfter({ kind: "fit" }, 0, 24)).toEqual({ kind: "fit" });
    expect(zoomAfter({ kind: "percent", value: 100 }, Number.NaN, 24)).toEqual({ kind: "percent", value: 100 });
    expect(zoomAfter({ kind: "percent", value: 100 }, -2, 24)).toEqual({ kind: "percent", value: 100 });
  });
});

describe("fitZoomExact", () => {
  it("is the number fitZoomPercent rounds", () => {
    expect(fitZoomExact(cell, photo, null)).toBeCloseTo(23.611, 2);
    expect(fitZoomPercent(cell, photo, null)).toBe(24);
  });
});

describe("showsWhole", () => {
  it("is true at Fit and at a fixed zoom no larger than the Fit zoom, which the core draws as Fit", () => {
    expect(showsWhole({ kind: "fit" }, null)).toBe(true);
    expect(showsWhole({ kind: "fit" }, 62)).toBe(true);
    expect(showsWhole({ kind: "percent", value: 50 }, 62)).toBe(true);
    expect(showsWhole({ kind: "percent", value: 62 }, 62)).toBe(true);
    expect(showsWhole({ kind: "percent", value: 62.5 }, 62)).toBe(false);
    expect(showsWhole({ kind: "percent", value: 100 }, 62)).toBe(false);
  });

  it("knows nothing about a fixed zoom before the cell has a size", () => {
    expect(showsWhole({ kind: "percent", value: 1 }, null)).toBe(false);
  });
});

describe("zoomAfter next to the Fit zoom", () => {
  it("starts a zoom in from the Fit zoom when a fixed zoom below it is shown as Fit", () => {
    // 50 % on a display where Fit is 62 % is Fit to the core, so one wheel step of 1.5 goes to 93 %, not 75 %.
    expect(zoomAfter({ kind: "percent", value: 50 }, 1.5, 62)).toEqual({ kind: "percent", value: 93 });
  });

  it("zooming out from such a zoom is Fit", () => {
    expect(zoomAfter({ kind: "percent", value: 50 }, 0.9, 62)).toEqual({ kind: "fit" });
  });

  it("a step that lands just above the Fit zoom is a fixed zoom, one that lands on it is Fit", () => {
    expect(zoomAfter({ kind: "percent", value: 100 }, 0.63, 62).kind).toBe("percent");
    expect(zoomAfter({ kind: "percent", value: 100 }, 0.62, 62)).toEqual({ kind: "fit" });
  });
});

describe("the menu steps", () => {
  const at = (value: number) => ({ kind: "percent", value }) as const;

  it("go through Fit, 50, 100 and 200 % from a place on the list", () => {
    expect(zoomStep({ kind: "fit" }, 1)).toEqual(at(50));
    expect(zoomStep(at(50), 1)).toEqual(at(100));
    expect(zoomStep(at(100), 1)).toEqual(at(200));
    expect(zoomStep(at(200), 1)).toEqual(at(200));
    expect(zoomStep(at(200), -1)).toEqual(at(100));
    expect(zoomStep(at(50), -1)).toEqual({ kind: "fit" });
    expect(zoomStep({ kind: "fit" }, -1)).toEqual({ kind: "fit" });
    expect(zoomStep(at(100), 0)).toEqual(at(100));
  });

  it("keep working from a zoom between the steps: up goes to the next step above, down to the next below", () => {
    expect(zoomKey(at(37.5))).toBe("37.5");
    expect(zoomStep(at(37.5), 1)).toEqual(at(50));
    expect(zoomStep(at(37.5), -1)).toEqual({ kind: "fit" });
    // A wheel zoom left the photo at 137 %: Zoom In must not jump back to 50 %.
    expect(zoomStep(at(137), 1)).toEqual(at(200));
    expect(zoomStep(at(137), -1)).toEqual(at(100));
    expect(zoomStep(at(137), 2)).toEqual(at(200));
    expect(zoomStep(at(137), -2)).toEqual(at(50));
    expect(zoomStep(at(350), 1)).toEqual(at(350));
    expect(zoomStep(at(350), -1)).toEqual(at(200));
  });

  it("skip the steps that would show Fit again, so a key press always does something", () => {
    // Fit is 62 % on a big display: 50 % is Fit, and Zoom In from Fit goes straight to 100 %.
    expect(zoomStep({ kind: "fit" }, 1, 62)).toEqual(at(100));
    expect(zoomStep(at(100), -1, 62)).toEqual({ kind: "fit" });
    expect(zoomStep(at(200), -1, 62)).toEqual(at(100));
    // A zoom the core shows as Fit counts as Fit.
    expect(zoomStep(at(50), 1, 62)).toEqual(at(100));
    expect(zoomStep(at(50), -1, 62)).toEqual({ kind: "fit" });
  });
});

describe("fitZoomOf", () => {
  it("takes the core's Fit zoom when a frame has reported one, else the estimate", () => {
    expect(fitZoomOf({ fit_percent: 31.2 }, 33)).toBe(31.2);
    expect(fitZoomOf({ fit_percent: null }, 33)).toBe(33);
    expect(fitZoomOf(null, 33)).toBe(33);
    expect(fitZoomOf(undefined, null)).toBeNull();
    // Nonsense from a sender is not a Fit zoom.
    expect(fitZoomOf({ fit_percent: 0 }, 33)).toBe(33);
    expect(fitZoomOf({ fit_percent: Number.NaN }, 33)).toBe(33);
    expect(fitZoomOf({ fit_percent: -4 }, null)).toBeNull();
  });
});

describe("panKeyDelta (the arrows of the zoomed photo, docs/28 A-10)", () => {
  const view = { width: 1000, height: 600 };

  it("an arrow moves the picture by 5 % of the view, the way a scroll does: → shows more of the right", () => {
    expect(PAN_STEP).toBe(0.05);
    expect(panKeyDelta("ArrowRight", false, view)).toEqual({ dx: -50, dy: 0 });
    expect(panKeyDelta("ArrowLeft", false, view)).toEqual({ dx: 50, dy: 0 });
    expect(panKeyDelta("ArrowDown", false, view)).toEqual({ dx: 0, dy: -30 });
    expect(panKeyDelta("ArrowUp", false, view)).toEqual({ dx: 0, dy: 30 });
  });

  it("with Shift it is 25 % of the view", () => {
    expect(PAN_STEP_SHIFT).toBe(0.25);
    expect(panKeyDelta("ArrowRight", true, view)).toEqual({ dx: -250, dy: 0 });
    expect(panKeyDelta("ArrowUp", true, view)).toEqual({ dx: 0, dy: 150 });
  });

  it("no other key is a pan (Home centres, I reads: they are not steps)", () => {
    for (const key of ["Home", "End", "i", "Enter", "a", " "]) expect(panKeyDelta(key, false, view)).toBeNull();
  });

  it("an empty view gives no step, and never a negative zero", () => {
    expect(panKeyDelta("ArrowRight", false, { width: 0, height: 0 })).toEqual({ dx: 0, dy: 0 });
    expect(Object.is(panKeyDelta("ArrowDown", false, { width: 0, height: 0 })?.dx, 0)).toBe(true);
  });

  it("a held key's steps are one pan of the sum (the queue the drag and the wheel use), and the core clamps it at the edge", async () => {
    // The page keeps no offset: it asks for a move and the core, which knows the picture's edges, clamps it (`ViewportState::input`, tested in
    // Rust: "panning moves the image with the pointer and stops at the edges"). So ten steps at a stop are the sum, and the sum is one command.
    const sent: StreamInput[] = [];
    let frame: (() => void) | null = null;
    const queue = new ViewportInputQueue({
      send: async (event) => void sent.push(event),
      mirrorZoom: () => {},
      onError: () => {},
      schedule: (callback) => (frame = callback),
    });
    for (let i = 0; i < 10; i++) {
      const step = panKeyDelta("ArrowRight", false, view);
      if (step) queue.push({ kind: "pan", ...step });
    }
    (frame as (() => void) | null)?.();
    await queue.flush();
    expect(sent).toEqual([{ kind: "pan", dx: -500, dy: 0 }]);
  });
});
