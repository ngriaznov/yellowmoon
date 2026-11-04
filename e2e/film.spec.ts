import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import type { ResolvedSettings } from "../src/api/types";
import { nodeOf } from "../src/state/settingsGraph";
import type { FilmItem } from "../src/state/settingsGraph";
import { openApp, openPhotos, recordCalls, setValue, watchErrors } from "./helpers/app";
import { filmView } from "./helpers/states";

// The Film node's interface (docs/24 section 4, task F-09), in a real browser on the mock API: the tabs, the Stock picker as tiles, the paper
// list, a slide stock's Print tab, the credit line and its links, About, the Output's note and Add node › Film. The page audit over the same
// states is in glass.spec.ts (the states 66 to 76 of helpers/states.ts).

type Mock = { getSettings(id: number): Promise<ResolvedSettings> };
const mock = <T>(page: Page, run: (api: Mock, ...args: never[]) => T | Promise<T>): Promise<T> =>
  page.evaluate(`(${run.toString()})(window.__yellowmoon.mockApi)`) as Promise<T>;
const filmNode = async (page: Page): Promise<FilmItem> => nodeOf((await mock(page, (api) => api.getSettings(4))).settings, 2) as FilmItem;

const panel = (page: Page) => page.getByRole("complementary", { name: "Tools" });
const tabNames = async (page: Page): Promise<string[]> => (await panel(page).getByRole("tab").allTextContents()).map((t) => t.trim());

test.describe("the Film node's tabs", () => {
  test("are Film, Print, Halation, Grain and Mask under a head that says Film · whole photo", async ({ page }) => {
    await filmView(page);
    const head = panel(page).getByRole("group", { name: "Node", exact: true });
    await expect(head.getByRole("heading", { level: 2, name: "Film" })).toBeVisible();
    await expect(head.locator(".ym-nodehead__line")).toHaveText("Film · whole photo");
    expect(await tabNames(page)).toEqual(["Film", "Print", "Halation", "Grain", "Mask"]);
    for (const name of ["Print", "Halation", "Grain", "Mask", "Film"]) {
      await panel(page).getByRole("tab", { name, exact: true }).click();
      await expect(panel(page).getByRole("tab", { name, exact: true })).toHaveAttribute("aria-selected", "true");
    }
  });

  test("have every control at its range and default", async ({ page }) => {
    await filmView(page);
    const slider = (name: string) => page.getByRole("slider", { name, exact: true });
    await expect(slider("Exposure")).toHaveAttribute("aria-valuemin", "-4");
    await expect(slider("Exposure")).toHaveAttribute("aria-valuemax", "4");
    await expect(slider("Strength")).toHaveAttribute("aria-valuenow", "100");
    await panel(page).getByRole("tab", { name: "Print", exact: true }).click();
    await expect(slider("Print exposure")).toHaveAttribute("aria-valuemin", "-3");
    await expect(slider("Yellow")).toHaveAttribute("aria-valuemax", "50");
    await expect(slider("Magenta")).toHaveAttribute("aria-valuenow", "0");
    await expect(page.getByText("0 is neutral for this film and paper.")).toBeVisible();
    await expect(slider("Cyan")).toHaveCount(0);
    await page.getByRole("region", { name: "Lights" }).getByRole("button", { name: "More" }).click();
    await expect(slider("Cyan")).toBeVisible();
    await panel(page).getByRole("tab", { name: "Halation", exact: true }).click();
    await expect(slider("Amount")).toHaveAttribute("aria-valuemax", "300");
    await expect(slider("Size")).toHaveAttribute("aria-valuemin", "50");
    await panel(page).getByRole("tab", { name: "Grain", exact: true }).click();
    await expect(slider("Amount")).toHaveAttribute("aria-valuemax", "200");
    await expect(slider("Size")).toHaveAttribute("aria-valuemax", "200");
    // No Brightness option: the owner chose "as the app" only.
    await panel(page).getByRole("tab", { name: "Print", exact: true }).click();
    await expect(page.getByText(/As printed/)).toHaveCount(0);
  });

  test("a value typed is one patch to the node, and the picture follows", async ({ page }) => {
    await filmView(page, "Halation");
    const calls = await recordCalls(page, "updateSettings");
    await setValue(page, "Amount", "180");
    await expect.poll(async () => (await filmNode(page)).halation.amount).toBe(180);
    expect(JSON.stringify((await calls()).at(-1)?.args[1])).toBe('{"nodes":{"2":{"halation":{"amount":180}}}}');
  });
});

test.describe("the Stock picker", () => {
  test("is six columns of tiles with no scrolling: 20 stocks, a small render of the photo in each, in three groups", async ({ page }) => {
    await filmView(page);
    await page.getByRole("button", { name: "Stock, Kodak Portra 400" }).click();
    const picker = page.getByRole("dialog", { name: "Film stocks" });
    await expect(picker).toBeVisible();
    await expect(picker.getByRole("heading", { level: 3 })).toHaveText(["Colour negative", "Cine negative", "Slide"]);
    await expect(picker.getByRole("button")).toHaveCount(20);
    await expect(picker.locator("img")).toHaveCount(20, { timeout: 20000 });
    // Six columns: the first row of a group has six tiles side by side.
    const firstRow = await picker.getByRole("region", { name: "Colour negative" }).getByRole("button").evaluateAll((tiles) => {
      const tops = tiles.map((t) => Math.round(t.getBoundingClientRect().top));
      return tops.filter((top) => top === tops[0]).length;
    });
    expect(firstRow).toBe(6);
    // No scrolling: the box shows all of its content.
    const scrolls = await picker.evaluate((el) => el.scrollHeight > el.clientHeight + 1);
    expect(scrolls).toBe(false);
    await expect(picker.getByRole("button", { name: "Kodak Portra 400, ISO 400" })).toHaveAttribute("aria-pressed", "true");
  });

  test("chooses a stock with a click, and shows its kind and ISO under the field", async ({ page }) => {
    await filmView(page);
    await page.getByRole("button", { name: "Stock, Kodak Portra 400" }).click();
    await page.getByRole("button", { name: "Kodak Vision3 500T, ISO 500" }).click();
    await expect(page.getByRole("dialog", { name: "Film stocks" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Stock, Kodak Vision3 500T" })).toBeVisible();
    await expect(page.getByText("Cine negative · ISO 500")).toBeVisible();
    expect((await filmNode(page)).film.stock).toBe("kodak_vision3_500t");
  });

  test("Esc closes it without a change", async ({ page }) => {
    await filmView(page);
    await page.getByRole("button", { name: "Stock, Kodak Portra 400" }).click();
    await expect(page.getByRole("dialog", { name: "Film stocks" })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "Film stocks" })).toHaveCount(0);
    expect((await filmNode(page)).film.stock).toBe("kodak_portra_400");
  });
});

test.describe("the paper list", () => {
  test("is Matching paper and the six papers, the matching paper's name under the field; the cine print films are not there", async ({ page }) => {
    await filmView(page, "Print");
    await expect(page.getByText("Kodak Professional Portra Endura")).toBeVisible();
    await page.getByRole("combobox", { name: "Paper" }).click();
    await expect(page.getByRole("option")).toHaveText([
      "Matching paper",
      "Fujifilm Crystal Archive Type II",
      "Kodak Ektacolor Edge",
      "Kodak Professional Endura Premier",
      "Kodak Professional Portra Endura",
      "Kodak Professional Supra Endura",
      "Kodak Professional Ultra Endura",
    ]);
    await page.getByRole("option", { name: "Fujifilm Crystal Archive Type II" }).click();
    await expect.poll(async () => (await filmNode(page)).print.paper).toBe("fujifilm_crystal_archive_typeii");
  });
});

test.describe("a slide", () => {
  test("has a Print tab that is dimmed and says there is no print, and can still be opened", async ({ page }) => {
    await filmView(page, "Print", { film: { stock: "fujifilm_velvia_100" } });
    await expect(page.getByText("Slide film is scanned as it is. There is no print.")).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Paper" })).toBeDisabled();
    await expect(page.getByRole("combobox", { name: "Paper" })).toHaveText("None");
    await expect(page.getByRole("slider", { name: "Yellow" })).toHaveAttribute("data-disabled", "");
    await expect(panel(page).getByRole("tab", { name: "Print", exact: true })).toBeEnabled();
    // The Film tab says Slide.
    await panel(page).getByRole("tab", { name: "Film", exact: true }).click();
    await expect(page.getByText("Slide · ISO 100")).toBeVisible();
  });
});

test.describe("the credit line", () => {
  test("ends the Film tab, with the author's name as a link to the project's page", async ({ page }) => {
    await filmView(page);
    const credit = panel(page).locator(".ym-film-credit");
    await expect(credit).toHaveText("Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm.");
    await credit.getByRole("button", { name: "Andrea Volpato (spektrafilm)" }).click();
    expect(await page.evaluate(() => (window as unknown as { __yellowmoon: { mockApi: { system: { opened: string[] } } } }).__yellowmoon.mockApi.system.opened)).toEqual([
      "https://github.com/andreavolpato/spektrafilm",
    ]);
  });

  test("is in About with the licence's link and the trademark line", async ({ page }) => {
    await filmView(page);
    await page.evaluate(() => window.__yellowmoon!.mockApi.emitEvent("menu_action", { action: "about" }));
    const about = page.getByRole("dialog", { name: "About Yellowmoon" });
    const film = about.getByRole("region", { name: "Film" });
    await expect(film).toContainText("Modelled on published data. Film profiles by Andrea Volpato (spektrafilm), CC BY-SA 4.0. Not affiliated with Kodak or Fujifilm.");
    await expect(film.getByRole("button", { name: "CC BY-SA 4.0" })).toBeVisible();
    await expect(film).toContainText("Kodak, Portra, Vision3, Fujifilm and Velvia are trademarks of their owners.");
  });
});

test.describe("the rest of the node", () => {
  test("the Output's Look tab carries the note while a Film node is live", async ({ page }) => {
    await filmView(page);
    await page.locator('[data-node-card="output"]').click();
    await expect(page.getByText("The look acts after the Film node. Choose Neutral to see the print as scanned.")).toBeVisible();
  });

  test("the Mask tab is a corrector's", async ({ page }) => {
    await filmView(page, "Mask");
    await expect(page.getByRole("region", { name: "Masks" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Add mask/ })).toBeVisible();
  });

  test("Add node › Film adds a Film node after the selected one and selects it", async ({ page }) => {
    const errors = watchErrors(page);
    await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
    await openPhotos(page);
    await page.getByRole("button", { name: "Nodes", exact: true }).click();
    await page.getByRole("button", { name: "Add node" }).click();
    await page.getByRole("menuitem", { name: /Film/ }).click();
    await expect(page.locator('[data-node-card="2"]')).toHaveAttribute("aria-pressed", "true");
    expect(await tabNames(page)).toEqual(["Film", "Print", "Halation", "Grain", "Mask"]);
    expect((await filmNode(page)).film).toEqual({ stock: "kodak_portra_400", exposure_ev: 0 });
    expect(errors.filter((e) => !e.includes("React Flow"))).toEqual([]);
  });
});
