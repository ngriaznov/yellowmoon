import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

// The component dev page (`components.html`, task M3-01) in a real browser: layout, pointer drags and focus rings of the components,
// on the same dark glass sheets as the app. The replicas of the screens (`?replica=`) open without errors; they are no longer compared
// pixel by pixel with `design/mockups` (the pre-glass design). The screens themselves are compared with `design/screenshots-c2/`
// (`screenshots.spec.ts`, `export.spec.ts`, `wizard.spec.ts`).

/** Collects console errors and page errors so a test can assert there were none. */
function watchErrors(page: Page): string[] {
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" || message.type() === "warning") problems.push(`${message.type()}: ${message.text()}`);
  });
  return problems;
}

async function ready(page: Page): Promise<void> {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(150);
}

test.describe("the dev page", () => {
  test("shows every component section without errors", async ({ page }) => {
    const problems = watchErrors(page);
    await page.goto("/components.html");
    await expect(page).toHaveTitle("Yellowmoon · Components");
    await ready(page);
    for (const name of ["Foundations", "Slider", "ValueBox", "Segmented", "Switch", "Card", "TabBar and TitleBlock", "LookTile", "Wheel", "WarpPlane", "CurvePlot and Histogram", "Filmstrip", "PhotoCell", "UpdatePill", "ExportQueueList and DevelopList", "DropZone", "StepRail"]) {
      await expect(page.getByRole("heading", { level: 2, name, exact: true })).toBeVisible();
    }
    expect(problems).toEqual([]);
  });

  test("uses Figtree and never scrolls sideways at 1440 px", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    await expect(page.locator("body")).toHaveCSS("font-family", /Figtree/);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test("draws the slider's parts at the sizes of docs/11", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const slider = page.locator("#slider [data-slider]").first();
    await expect(slider).toHaveCSS("height", "38px");
    const geometry = await slider.evaluate((el) => {
      const box = (selector: string) => {
        const r = (el.querySelector(selector) as HTMLElement).getBoundingClientRect();
        return { w: Math.round(r.width * 100) / 100, h: Math.round(r.height * 100) / 100 };
      };
      return { track: box('[data-slot="slider-track"]'), knob: box('[data-slot="slider-thumb"]'), value: box('[data-slot="input"]') };
    });
    // glasscn's slider: a 6 px soft track and its small round 16 px knob (`size="sm"`, the owner's request of 2026-10-17: the 36 × 24
    // capsule took a third of the travel); the box is its small input.
    expect(geometry.track.h).toBe(6);
    expect(geometry.knob).toEqual({ w: 16, h: 16 });
    expect(geometry.value.w).toBe(72);
    expect(geometry.value.h).toBeLessThanOrEqual(38);
    const columns = await slider.evaluate((el) => getComputedStyle(el).gridTemplateColumns.split(" ").map((c) => Math.round(parseFloat(c))));
    expect(columns[0]).toBe(92);
    expect(columns[2]).toBe(72);
    expect(columns[3]).toBe(40);
  });
});

test.describe("slider in a real browser", () => {
  test("drags with the pointer, reports one patch per frame and a final one, and double-click resets", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Live · exposure with Auto" });
    const control = demo.locator('[data-slot="slider"]');
    await control.scrollIntoViewIfNeeded();
    const box = (await control.boundingBox())!;
    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + box.width * 0.5, y);
    await page.mouse.down();
    for (let i = 1; i <= 10; i++) await page.mouse.move(box.x + box.width * (0.5 + i * 0.03), y);
    await page.mouse.up();
    const log = (await demo.locator(".dev-output").textContent()) ?? "";
    expect(log).toMatch(/drag/);
    expect(log).toMatch(/final/);
    const lines = log.trim().split("\n");
    expect(lines.at(-1)).toMatch(/^final/);
    // Ten moves arrive in far fewer than ten frames' worth of patches.
    expect(lines.filter((l) => l.startsWith("drag")).length).toBeLessThanOrEqual(10);
    await expect(demo.getByRole("textbox", { name: "Exposure value" })).not.toHaveValue("+0.30");
    await control.dblclick();
    await expect(demo.getByRole("textbox", { name: "Exposure value" })).toHaveValue("+0.30");
    await expect(demo.locator(".dev-output")).toContainText("reset to Auto");
  });

  test("puts the thumb under the pointer", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Intensity · default at the right end" });
    const control = demo.locator('[data-slot="slider"]');
    await control.scrollIntoViewIfNeeded();
    const box = (await control.boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.25, box.y + box.height / 2);
    await expect(demo.getByRole("textbox", { name: "Intensity value" })).toHaveValue("25%");
    await expect(demo.getByRole("slider", { name: "Intensity" })).toHaveAttribute("aria-valuenow", "25");
    // Radix keeps the knob inside the track, so it sits a few pixels off the pointer at the quarter point (half its width less a quarter of it).
    const knob = (await demo.locator('[data-slot="slider-thumb"]').boundingBox())!;
    expect(Math.abs(knob.x + knob.width / 2 - (box.x + box.width * 0.25))).toBeLessThan(knob.width / 2);
  });

  test("shows a 3 px focus ring on the slider's knob when reached with the keyboard, and steps with the arrows", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Intensity · default at the right end" });
    const input = demo.getByRole("slider", { name: "Intensity" });
    await input.focus();
    await page.keyboard.press("Shift+ArrowLeft");
    await expect(demo.getByRole("textbox", { name: "Intensity value" })).toHaveValue("90%");
    await page.keyboard.press("Home");
    await expect(demo.getByRole("textbox", { name: "Intensity value" })).toHaveValue("0%");
    // Reach it by Tab so :focus-visible applies.
    await page.locator("body").click({ position: { x: 5, y: 5 } });
    await demo.getByRole("textbox", { name: "Intensity value" }).focus();
    await page.keyboard.press("Shift+Tab");
    await expect(input).toBeFocused();
    // The knob is the focus stop and draws the ring itself.
    await expect(input).toHaveCSS("outline-style", "solid");
    await expect(input).toHaveCSS("outline-width", "3px");
    await expect(input).toHaveCSS("outline-offset", "2px");
  });

  test("the value box takes a typed value, clamps and commits on Return", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Live · kelvin" });
    const box = demo.getByRole("textbox", { name: "Kelvin value" });
    await box.click();
    await page.keyboard.press("ControlOrMeta+a");
    await page.keyboard.type("99999");
    await page.keyboard.press("Enter");
    await expect(box).toHaveValue("11000 K");
    await box.click();
    await page.keyboard.press("ControlOrMeta+a");
    await page.keyboard.type("3000");
    await page.keyboard.press("Escape");
    await expect(box).toHaveValue("11000 K");
  });
});

test.describe("wheel and curve in a real browser", () => {
  test("a wheel follows the pointer, stays inside its disc and resets on a double click", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Live · three wheels" });
    const wheel = demo.getByRole("slider", { name: "Midtones colour balance" });
    await wheel.scrollIntoViewIfNeeded();
    const box = (await wheel.boundingBox())!;
    expect(Math.round(box.width)).toBe(92);
    expect(Math.round(box.height)).toBe(92);
    // A small round point (10 px, the owner's request of 2026-10-17), in the very middle of the disc while centred.
    const rest = (await wheel.locator(".ym-wheel__puck").boundingBox())!;
    expect([rest.width, rest.height]).toEqual([10, 10]);
    expect(Math.abs(rest.x + rest.width / 2 - (box.x + box.width / 2))).toBeLessThanOrEqual(0.5);
    expect(Math.abs(rest.y + rest.height / 2 - (box.y + box.height / 2))).toBeLessThanOrEqual(0.5);
    await page.mouse.move(box.x + 46, box.y + 46);
    await page.mouse.down();
    await page.mouse.move(box.x + 46 + 200, box.y + 46, { steps: 4 });
    await page.mouse.up();
    // Dragged far past the rim: the point is pulled back to it, to the right of centre.
    await expect(wheel).toHaveAttribute("aria-valuenow", "100");
    const puck = (await wheel.locator(".ym-wheel__puck").boundingBox())!;
    expect(puck.x + puck.width / 2).toBeGreaterThan(box.x + 46);
    expect(puck.x + puck.width / 2).toBeLessThan(box.x + box.width);
    await wheel.dblclick();
    await expect(wheel).toHaveAttribute("aria-valuetext", "Centred");
  });

  test("a curve adds a point by pressing, moves it by dragging and deletes it by dragging off the plot", async ({ page }) => {
    await page.goto("/components.html?replica=photos&tab=curve");
    await ready(page);
    const plot = page.getByRole("group", { name: "Luma curve" });
    const before = await plot.getByRole("slider").count();
    const box = (await plot.boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.12, box.y + box.height * 0.7);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.12, box.y + box.height * 0.5, { steps: 3 });
    await page.mouse.up();
    await expect(plot.getByRole("slider")).toHaveCount(before + 1);
    // Drag the same point off the bottom of the plot.
    await page.mouse.move(box.x + box.width * 0.12, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.12, box.y + box.height + 60, { steps: 4 });
    await page.mouse.up();
    await expect(plot.getByRole("slider")).toHaveCount(before);
  });

  test("a curve is keyboard-operable: Return adds a point, arrows move it, Delete removes it", async ({ page }) => {
    await page.goto("/components.html");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Identity (two end points)" });
    const plot = demo.getByRole("group", { name: "Luma curve" });
    await plot.scrollIntoViewIfNeeded();
    await plot.focus();
    await page.keyboard.press("Enter");
    const points = plot.getByRole("slider");
    await expect(points).toHaveCount(3);
    await expect(points.nth(1)).toBeFocused();
    await page.keyboard.press("ArrowUp");
    await expect(points.nth(1)).toHaveAttribute("aria-valuetext", "Input 128, output 129");
    await page.keyboard.press("Delete");
    await expect(points).toHaveCount(2);
  });
});

test.describe("photo cell in a real browser", () => {
  for (const size of [{ width: 1440, height: 900 }, { width: 1180, height: 760 }, { width: 1800, height: 1100 }]) {
    test(`is exactly 3:2 and reports its rectangle at ${size.width} × ${size.height}`, async ({ page }) => {
      await page.setViewportSize(size);
      await page.goto("/components.html?replica=photos");
      await ready(page);
      const cell = page.locator("[data-viewport]");
      const box = (await cell.boundingBox())!;
      expect(box.width / box.height).toBeCloseTo(1.5, 2);
      // width: min(100%, (max(100vh, 760px) − 199px) · 1.5)
      const column = size.width - 400;
      const expected = Math.min(column, (Math.max(size.height, 760) - 199) * 1.5);
      expect(box.width).toBeCloseTo(expected, 0);
      // Centred in its column.
      expect(box.x).toBeCloseTo((column - expected) / 2, 0);
      // No hairline gap above or below: the filmstrip's rule starts where the cell ends.
      const strip = (await page.getByRole("navigation", { name: "Photos" }).boundingBox())!;
      expect(strip.y).toBeCloseTo(box.y + box.height, 0);
    });
  }

  test("reports its rectangle again when the window resizes", async ({ page }) => {
    await page.goto("/components.html#photo");
    await ready(page);
    const readout = page.locator("figure.dev-demo", { hasText: "Look chip and camera line" }).locator(".dev-output");
    await expect(readout).toContainText("onRect: x");
    const first = await readout.textContent();
    await page.setViewportSize({ width: 1200, height: 800 });
    await page.waitForTimeout(200);
    const second = await readout.textContent();
    expect(second).not.toBe(first);
  });
});

test.describe("shortcuts and drops in a real browser", () => {
  test("⌘1–⌘6 switch tabs from anywhere, even inside a value box", async ({ page }) => {
    await page.goto("/components.html?replica=photos&tab=look");
    await ready(page);
    await page.getByRole("textbox", { name: "Intensity value" }).focus();
    await page.keyboard.press("Meta+Digit3");
    await expect(page.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Meta+Digit6");
    await expect(page.getByRole("tab", { name: "RAW" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Meta+Digit1");
    await expect(page.getByRole("tab", { name: "Look" })).toHaveAttribute("aria-selected", "true");
  });

  test("← and → move between tabs inside the bar", async ({ page }) => {
    await page.goto("/components.html?replica=photos&tab=look");
    await ready(page);
    await page.getByRole("tab", { name: "Look" }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Light" })).toBeFocused();
    await expect(page.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("tab", { name: "Look" })).toBeFocused();
    // From the first tab the arrow wraps to the last.
    await page.keyboard.press("ArrowLeft");
    await expect(page.getByRole("tab", { name: "RAW" })).toBeFocused();
  });

  test("files dropped on the window and on the zone are reported", async ({ page }) => {
    await page.goto("/components.html#drop");
    await ready(page);
    const demo = page.locator("figure.dev-demo", { hasText: "Compact (wizard)" });
    const zone = demo.getByRole("group", { name: "Or drop the zip here" });
    await zone.scrollIntoViewIfNeeded();
    const dispatch = async (target: "zone" | "window", names: string[]) =>
      page.evaluate(
        ({ target, names }) => {
          const el = target === "zone" ? document.querySelector('[aria-label="Or drop the zip here"]')! : document.body;
          const dt = new DataTransfer();
          for (const name of names) dt.items.add(new File(["x"], name));
          for (const type of ["dragenter", "dragover", "drop"]) el.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
        },
        { target, names },
      );
    await dispatch("zone", ["DSC04201.ARW", "DSC04202.ARW"]);
    await expect(demo.locator(".dev-output")).toContainText("zone drop: DSC04201.ARW, DSC04202.ARW");
    await expect(demo.locator(".dev-output")).toContainText("window drop: 2 to open, 0 ignored");
    await dispatch("window", ["notes.txt", "photo.jpg"]);
    await expect(demo.locator(".dev-output")).toContainText("window drop: 0 to open, 2 ignored");
    await expect(demo.locator(".dev-output")).toContainText("No ARW files in what you dropped.");
  });

  test("focus rings are 3 px: outside a switch (2 px offset), inside a tab and a segment (inset)", async ({ page }) => {
    await page.goto("/components.html?replica=photos&tab=raw");
    await ready(page);
    await page.keyboard.press("Tab");
    const ring = async (locator: ReturnType<Page["getByRole"]>, offset: string): Promise<void> => {
      await locator.focus();
      await page.keyboard.press("Shift+Tab");
      await page.keyboard.press("Tab");
      await expect(locator).toBeFocused();
      await expect(locator).toHaveCSS("outline-width", "3px");
      await expect(locator).toHaveCSS("outline-style", "solid");
      await expect(locator).toHaveCSS("outline-offset", offset);
    };
    await ring(page.getByRole("switch", { name: "Neural demosaic" }), "2px");
    // The tab row and a segmented control are panes of their own: their ring is drawn inside the item.
    await ring(page.getByRole("tab", { name: "RAW" }), "-3px");
    await page.goto("/components.html?replica=photos&tab=color");
    await ready(page);
    await page.getByRole("radio", { name: "Auto" }).focus();
    await page.keyboard.press("ArrowRight");
    const radio = page.getByRole("radio", { name: "Day" });
    await expect(radio).toBeFocused();
    await expect(radio).toHaveCSS("outline-width", "3px");
    await expect(radio).toHaveCSS("outline-offset", "-3px");
  });
});

test.describe("the replicas of the screens", () => {
  // The screens of design/mockups rebuilt from the components (`?replica=`). They are the dev page's own scaffolding: each opens and
  // draws without a script error. They were compared pixel by pixel with the mockups until the dark glass interface (2026-10-16); the
  // mockups are the pre-glass design now, and the real screens are compared with `design/screenshots-c2/` instead.
  const queries = [
    "replica=photos&tab=look",
    "replica=photos&tab=light",
    "replica=photos&tab=color",
    "replica=photos&tab=detail",
    "replica=photos&tab=curve",
    "replica=photos&tab=raw",
    "replica=photos&tab=look&before=1",
    "replica=photos&tab=look&q=film",
    "replica=start",
    "replica=start&prog=5.4",
    "replica=start&prog=12",
    "replica=export&fmt=jpeg",
    "replica=export&fmt=tiff",
    "replica=export&fmt=png",
    "replica=export&fmt=dng",
    "replica=export&running=1.45",
    "replica=export&running=3",
    "replica=wizard&step=1",
    "replica=wizard&step=3&imp=40",
  ];
  for (const query of queries) {
    test(`${query} opens without errors and fills the window`, async ({ page }) => {
      const problems = watchErrors(page);
      await page.goto(`/components.html?${query}`);
      await ready(page);
      const screen = page.locator(".rp-screen").first();
      await expect(screen).toBeVisible();
      const box = (await screen.boundingBox())!;
      expect(box.width).toBeGreaterThan(1000);
      expect(box.height).toBeGreaterThan(600);
      expect(problems).toEqual([]);
    });
  }
});
