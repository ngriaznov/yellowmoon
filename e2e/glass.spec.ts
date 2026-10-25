import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { auditPage } from "./helpers/audit";
import { openApp, openPhotos, watchErrors } from "./helpers/app";
import { openState, STATES } from "./helpers/states";

// The dark glass interface (C2, the owner's decision of 2026-10-16, docs/11): every control is a glasscn component (Radix underneath),
// and it behaves as a native one does. These are the checks the interface was built against, run in a real browser on the mock API:
// the page audit over every state, and the keys, the pointer and the ARIA state of the controls.

const MOCKUP = { mockCurrent: 3, mockPicked: "2,3,4" } as const;

async function photos(page: Page): Promise<void> {
  await openApp(page, MOCKUP);
  await openPhotos(page);
}

const box = (page: Page, label: string) => page.getByRole("textbox", { name: `${label} value` });

test.describe("the page audit: no element of ours where glasscn has one", () => {
  // Every visible control of a page is a glasscn component, and so is every row, surface and badge that glasscn has a counterpart for.
  // What is left is the picture-like and layout-like parts that glasscn has nothing for (the wheels, the curve, the histogram, the look
  // tiles, the photo cell and the bands of the tool panel): they are listed by the audit, not hidden.
  for (const [name] of STATES) {
    test(name, async ({ browser }) => {
      const { page, context } = await openState(browser, name);
      try {
        const audit = await auditPage(page);
        const left = [...audit.ours.map((o) => `control: ${o}`), ...audit.pending.map((p) => `surface: ${p}`)];
        expect(left, `elements of ours with a glasscn counterpart on ${name}`).toEqual([]);
        // A page that shows controls shows glasscn's.
        expect(Object.values(audit.glass).reduce((a, b) => a + b, 0), "glasscn controls on the page").toBeGreaterThan(0);
      } finally {
        await context.close();
      }
    });
  }
});

test.describe("tabs", () => {
  test("← and → move and select, ⌘4 selects Colour, and the panel is labelled by its tab", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Look" }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Lens" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Light" })).toHaveAttribute("aria-selected", "true");
    await page.keyboard.press("Meta+4");
    await expect(page.getByRole("tab", { name: "Colour" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("tabpanel")).toHaveAttribute("aria-labelledby", "ym-tab-color");
  });

  test("the labels sit in the middle of the tab row and the underline on its rim", async ({ page }) => {
    // The owner's report of 2026-10-17: the labels sat in the row's top 32 px and the underline floated 13 px above the rim.
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const g = await page.locator(".ym-tools").evaluate((tools) => {
      const rect = (el: Element | null) => el!.getBoundingClientRect();
      const row = rect(tools.querySelector(':scope > [data-slot="tabs"]'));
      const label = (() => {
        const range = document.createRange();
        range.selectNodeContents(tools.querySelector('[data-slot="tabs-trigger"][data-state="active"]')!);
        return range.getBoundingClientRect();
      })();
      const line = rect(tools.querySelector('[data-slot="tabs-indicator"]'));
      const rim = parseFloat(getComputedStyle(tools.querySelector(':scope > [data-slot="tabs"]')!).borderBottomWidth);
      return { rowMid: (row.top + row.bottom - rim) / 2, labelMid: (label.top + label.bottom) / 2, lineBottom: line.bottom, rimTop: row.bottom - rim };
    });
    expect(Math.abs(g.labelMid - g.rowMid), JSON.stringify(g)).toBeLessThanOrEqual(1);
    expect(Math.abs(g.lineBottom - g.rimTop), JSON.stringify(g)).toBeLessThanOrEqual(0.5);
  });
});

test.describe("the tab row at three window sizes", () => {
  // Seven labels in the 400 px panel's 46 px row (the owner's request of 2026-10-25: Lens is the second tab). Each label is on one line with
  // its underline under it, nothing is clipped or overlaps its neighbour, whatever the window: the panel is 400 px at every size, so the
  // three sizes also show that nothing in the row depends on the window.
  for (const [width, height] of [
    [1440, 900],
    [1180, 760],
    [1800, 1100],
  ] as const) {
    test(`every label fits on one line with its underline, at ${width} × ${height}`, async ({ page }) => {
      await page.setViewportSize({ width, height });
      await photos(page);
      const ids = ["look", "lens", "light", "color", "detail", "curve", "raw"];
      for (const id of ids) {
        await page.locator(`#ym-tab-${id}`).click();
        await expect(page.locator(`#ym-tab-${id}`)).toHaveAttribute("aria-selected", "true");
        // The underline slides to the chosen tab; measure it where it has stopped.
        await expect
          .poll(async () =>
            page.locator(".ym-tools").evaluate((tools, tab) => {
              const line = tools.querySelector('[data-slot="tabs-indicator"]')!.getBoundingClientRect();
              const trigger = tools.querySelector(`#ym-tab-${tab}`)!.getBoundingClientRect();
              return Math.max(Math.abs(line.left - trigger.left), Math.abs(line.right - trigger.right));
            }, id),
          )
          .toBeLessThanOrEqual(1);
      }
      const g = await page.locator(".ym-tools").evaluate((tools) => {
        const row = tools.querySelector(':scope > [data-slot="tabs"]')!.getBoundingClientRect();
        const triggers = [...tools.querySelectorAll<HTMLElement>('[data-slot="tabs-trigger"]')];
        const labels = triggers.map((trigger) => {
          const range = document.createRange();
          range.selectNodeContents(trigger);
          const rects = [...range.getClientRects()];
          const box = trigger.getBoundingClientRect();
          return {
            name: trigger.textContent,
            lines: new Set(rects.map((r) => Math.round(r.top))).size,
            left: Math.min(...rects.map((r) => r.left)),
            right: Math.max(...rects.map((r) => r.right)),
            cell: { left: box.left, right: box.right },
            overflow: trigger.scrollWidth - trigger.clientWidth,
          };
        });
        return { rowHeight: row.height, rowLeft: row.left, rowRight: row.right, labels };
      });
      expect(g.rowHeight, JSON.stringify(g)).toBe(46);
      expect(g.labels.map((l) => l.name)).toEqual(["Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW"]);
      for (const [i, l] of g.labels.entries()) {
        expect(l.lines, `${l.name} is on one line`).toBe(1);
        expect(l.overflow, `${l.name} overflows its cell`).toBeLessThanOrEqual(0);
        // The label sits inside its own cell (so the underline, which is the cell's width, runs under all of it) and inside the row.
        expect(l.left, `${l.name} starts inside its cell`).toBeGreaterThanOrEqual(l.cell.left - 0.5);
        expect(l.right, `${l.name} ends inside its cell`).toBeLessThanOrEqual(l.cell.right + 0.5);
        expect(l.left, `${l.name} is not clipped on the left`).toBeGreaterThanOrEqual(g.rowLeft + 4);
        expect(l.right, `${l.name} is not clipped on the right`).toBeLessThanOrEqual(g.rowRight - 4);
        const next = g.labels[i + 1];
        if (next) expect(next.left - l.right, `the gap between ${l.name} and ${next.name}`).toBeGreaterThanOrEqual(8);
      }
    });
  }
});

test.describe("sliders and their boxes", () => {
  test("a slider is named, reads its value aloud, and steps with the arrows, Shift and Home", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const slider = page.getByRole("slider", { name: "Exposure" });
    await expect(slider).toHaveAttribute("aria-valuetext", /[+−]?\d/);
    await slider.focus();
    const start = await box(page, "Exposure").inputValue();
    await page.keyboard.press("ArrowRight");
    const step = await box(page, "Exposure").inputValue();
    await page.keyboard.press("Shift+ArrowRight");
    const tenSteps = await box(page, "Exposure").inputValue();
    await page.keyboard.press("Home");
    const home = await box(page, "Exposure").inputValue();
    expect(new Set([start, step, tenSteps, home]).size, `the box read ${[start, step, tenSteps, home].join(", ")}`).toBe(4);
    expect(parseFloat(tenSteps.replace("−", "-")) - parseFloat(step.replace("−", "-"))).toBeCloseTo(0.1, 2);
  });

  test("a drag moves the value and a double click on the track goes back to Auto", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const knob = page.getByRole("slider", { name: "Exposure" });
    const start = await box(page, "Exposure").inputValue();
    const at = (await knob.boundingBox())!;
    await page.mouse.move(at.x + at.width / 2, at.y + at.height / 2);
    await page.mouse.down();
    await page.mouse.move(at.x + at.width / 2 + 40, at.y + at.height / 2, { steps: 5 });
    await page.mouse.up();
    await expect(box(page, "Exposure")).not.toHaveValue(start);
    const track = page.locator('[data-slot="slider"]').first();
    const tb = (await track.boundingBox())!;
    await page.mouse.dblclick(tb.x + tb.width * 0.2, tb.y + tb.height / 2);
    await expect(box(page, "Exposure")).toHaveValue(start);
  });

  test("a typed value commits with Return and is dropped with Esc", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const contrast = box(page, "Contrast");
    await contrast.click();
    await contrast.fill("25");
    await contrast.press("Enter");
    await expect(contrast).toHaveValue("+25");
    await contrast.click();
    await contrast.fill("90");
    await contrast.press("Escape");
    await expect(contrast).toHaveValue("+25");
    await expect(page.getByRole("button", { name: "Reset Contrast" })).toHaveCount(1);
  });

  test("the AUTO toggle is pressed while a card is automatic, and pressing it after a change restores Auto", async ({ page }) => {
    await photos(page);
    await page.getByRole("tab", { name: "Light" }).click();
    const auto = page.getByRole("region", { name: "Exposure" }).getByRole("button", { name: "Auto", exact: true });
    await expect(auto).toHaveAttribute("aria-pressed", "true");
    await page.getByRole("slider", { name: "Exposure" }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(auto).toHaveAttribute("aria-pressed", "false");
    await auto.click();
    await expect(auto).toHaveAttribute("aria-pressed", "true");
  });

  test("the Crop panel's AUTO is a glasscn Toggle that switches both ways (Auto crop, docs/20)", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Crop", exact: true }).click();
    const card = page.getByRole("region", { name: "Frame" });
    const auto = card.getByRole("button", { name: "Auto crop" });
    await expect(auto).toHaveAttribute("data-slot", "toggle");
    await expect(auto).toHaveText("Auto");
    await expect(auto).toHaveAttribute("aria-pressed", "false");
    await auto.click();
    await expect(auto).toHaveAttribute("aria-pressed", "true");
    await expect(card.getByText("Cropped to frame the subject.")).toBeVisible();
    await expect(card.getByText("Levelled by 1.4°.")).toBeVisible();
    await auto.click();
    await expect(auto).toHaveAttribute("aria-pressed", "false");
    await expect(card.getByText("Cropped to frame the subject.")).toHaveCount(0);
  });
});

test.describe("the other controls", () => {
  test("a segmented control is a radio group and the arrows move its selection", async ({ page }) => {
    await photos(page);
    const group = page.getByRole("radiogroup", { name: "Apply look to" });
    await expect(group.getByRole("radio")).toHaveCount(3);
    await group.getByRole("radio", { name: "This photo" }).click();
    await page.keyboard.press("ArrowRight");
    await expect(group.getByRole("radio", { checked: true })).toContainText("Picked");
  });

  test("the search field is named and filters the tiles", async ({ page }) => {
    await photos(page);
    const search = page.getByRole("searchbox", { name: "Search looks" });
    const before = await page.locator(".ym-look-tile").count();
    await search.fill("signature");
    await expect.poll(() => page.locator(".ym-look-tile").count()).toBeLessThan(before);
    await search.fill("");
    await expect.poll(() => page.locator(".ym-look-tile").count()).toBe(before);
  });

  test("a title is a glass tooltip on hover, and a locked button explains itself too", async ({ page }) => {
    await photos(page);
    await page.getByRole("button", { name: "Before" }).hover();
    await expect(page.getByRole("tooltip").first()).toBeVisible();
    // The Export cell of Start is locked until there are photos; its tooltip says why.
    await page.goto("/?mockSpeed=Infinity&mockLooks=0");
    // A locked button takes no pointer events itself (aria-disabled), so the pointer goes to the tooltip's trigger around it.
    const locked = page.getByRole("button", { name: /^Export/ });
    await expect(locked).toHaveAttribute("aria-disabled", "true");
    await page.locator('[data-slot="tooltip-trigger"]', { has: locked }).hover();
    await expect(page.getByRole("tooltip").first()).toContainText("Open photos to export");
  });

  test("the zoom menu opens, lists its sizes, and Esc gives the focus back to the button", async ({ page }) => {
    await photos(page);
    const zoom = page.getByRole("button", { name: /^(Fit|\d+%)/ }).first();
    await zoom.click();
    await expect(page.getByRole("menu", { name: "Zoom" })).toBeVisible();
    await expect(page.getByRole("menuitemradio")).toHaveCount(4);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("menu")).toHaveCount(0);
    await expect(zoom).toBeFocused();
  });
});

test.describe("the header", () => {
  test("Before, Pick white and Crop are toggles: aria-pressed is false, true, false as they are pressed", async ({ page }) => {
    await photos(page);
    for (const name of ["Before", "Pick white", "Crop"]) {
      const toggle = page.getByRole("button", { name, exact: true });
      await expect(toggle, `${name} before`).toHaveAttribute("aria-pressed", "false");
      await toggle.click();
      await expect(toggle, `${name} on`).toHaveAttribute("aria-pressed", "true");
      await toggle.click();
      await expect(toggle, `${name} off`).toHaveAttribute("aria-pressed", "false");
    }
  });

  test("the toggles are glasscn Toggles, the actions glass buttons, and the status and the Actions landmark are there", async ({ page }) => {
    await photos(page);
    await expect(page.locator('.ym-header [data-slot="toggle"]')).toHaveCount(3);
    await expect(page.getByRole("navigation", { name: "Actions" })).toBeVisible();
    await expect(page.locator(".ym-status").first()).toContainText("·");
    const slots = await page.locator(".ym-header [data-slot]").evaluateAll((els) => [...new Set(els.map((e) => e.getAttribute("data-slot")))].sort());
    expect(slots).toEqual(expect.arrayContaining(["button", "toggle"]));
  });
});

test.describe("the filmstrip", () => {
  test("has no counter cell; one caption line follows the selection and the picks, and describes the strip", async ({ page }) => {
    await photos(page);
    await expect(page.locator(".ym-filmstrip__counter")).toHaveCount(0);
    const caption = page.locator(".ym-filmstrip__caption");
    await expect(caption).toHaveText("4 of 12 · 3 picked");
    const [c, thumb, strip, scroll] = await Promise.all([
      caption.boundingBox(),
      page.locator(".ym-filmstrip__thumb").first().boundingBox(),
      page.locator(".ym-filmstrip").boundingBox(),
      page.locator(".ym-filmstrip__scroll").boundingBox(),
    ]);
    // The caption sits in the strip's top inset, above the thumbnails, and the strip runs the full width of its row.
    expect(c!.y).toBeGreaterThanOrEqual(strip!.y + 1);
    expect(c!.y + c!.height).toBeLessThanOrEqual(thumb!.y);
    expect(Math.abs(scroll!.x + scroll!.width - (strip!.x + strip!.width))).toBeLessThan(0.5);
    await expect(page.getByRole("navigation", { name: "Photos" })).toHaveAccessibleDescription("4 of 12 · 3 picked");
    await page.getByRole("button", { name: /^Show DSC04204/ }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(caption).toHaveText("5 of 12 · 3 picked");
    await page.keyboard.press("ArrowLeft");
    await expect(caption).toHaveText("4 of 12 · 3 picked");
    await page.keyboard.press("p");
    await expect(caption).toHaveText("4 of 12 · 2 picked");
    await page.keyboard.press("p");
    await expect(caption).toHaveText("4 of 12 · 3 picked");
  });
});

test.describe("dialogs", () => {
  test("Preferences: role dialog, Done has the focus, Tab stays inside, Esc closes it and the focus goes back", async ({ page }) => {
    const problems = watchErrors(page);
    await photos(page);
    await page.getByRole("button", { name: /^Fit|\d+%/ }).first().focus();
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    await expect(dialog.getByRole("button", { name: "Done" })).toBeFocused();
    for (let i = 0; i < 14; i++) await page.keyboard.press("Tab");
    expect(await page.evaluate(() => !!document.activeElement?.closest('[role="dialog"]'))).toBe(true);
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^Fit|\d+%/ }).first()).toBeFocused();
    expect(problems).toEqual([]);
  });

  test("a switch toggles aria-checked with Space, and the Default look select opens a list of looks", async ({ page }) => {
    await photos(page);
    await page.keyboard.press("Meta+,");
    const dialog = page.getByRole("dialog", { name: "Preferences" });
    await expect(dialog).toBeVisible();
    const location = dialog.getByRole("switch", { name: "Remove location from exports" });
    await expect(location).toHaveAttribute("aria-checked", "false");
    await location.focus();
    await page.keyboard.press("Space");
    await expect(location).toHaveAttribute("aria-checked", "true");
    await dialog.getByRole("combobox", { name: "Default look" }).click();
    expect(await page.getByRole("option").count()).toBeGreaterThan(3);
    await page.keyboard.press("Escape");
    await expect(page.getByRole("option")).toHaveCount(0);
  });
});
