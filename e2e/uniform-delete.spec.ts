import { expect, test } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

// The owner's sequence of 0.2.4-beta.3, on the mock (CU-fix): Add node › Colour Uniform, Pick colour off and on, a click on the photo, a slider, the
// node's Delete, then a slider on Main. The test keeps the log of what the page asks of the core and checks the order the shell needs: the pick is one
// reading and one patch, no mask view is sent for a node that is not there, and the edit of Main after the delete reaches the core.

test("add, pick, change, delete and edit Main: every edit reaches the core, and no mask view names the deleted node", async ({ page }) => {
  await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
  await openPhotos(page);
  await page.evaluate(() => {
    const api = window.__yellowmoon!.mockApi as unknown as Record<string, (...a: unknown[]) => unknown>;
    const log: string[] = [];
    (window as unknown as { __log: string[] }).__log = log;
    for (const name of ["viewportSetMode", "updateSettings", "sampleChroma"]) {
      const call = api[name]!;
      api[name] = (...a: unknown[]) => {
        log.push(`${name} ${JSON.stringify(name === "viewportSetMode" ? (a[1] ?? null) : a.slice(1))}`);
        return call.apply(api, a);
      };
    }
  });
  const log = (): Promise<string[]> => page.evaluate(() => (window as unknown as { __log: string[] }).__log);
  await page.getByRole("tab", { name: "Light", exact: true }).click();
  await page.getByRole("button", { name: "Nodes", exact: true }).click();
  await page.getByRole("button", { name: "Add node" }).click();
  await page.getByRole("menu", { name: "Add node" }).getByRole("menuitem", { name: /^Colour Uniform/ }).click();
  const tools = page.getByRole("complementary", { name: "Tools" });
  const pick = tools.getByRole("region", { name: "Target" }).getByRole("button", { name: "Pick colour" });
  await expect(pick).toHaveAttribute("aria-pressed", "true");
  await pick.click();
  await pick.click();
  const layer = page.getByRole("application", { name: /Pick colour for the target of Colour Uniform/ });
  const box = (await layer.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.3);
  await expect.poll(async () => (await log()).filter((c) => c.startsWith("sampleChroma")).length).toBe(1);
  const hue = tools.getByRole("region", { name: "Compress" }).getByRole("slider", { name: "Hue" }).first();
  await expect(hue).not.toHaveAttribute("data-disabled");
  await hue.focus();
  await page.keyboard.press("ArrowRight");
  await tools.getByRole("button", { name: /more/i }).first().click();
  await page.getByRole("menuitem", { name: "Delete node" }).click();
  await expect(page.getByRole("button", { name: /^Node 2, Uniform/ })).toHaveCount(0);
  const slider = tools.getByRole("slider").first();
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(async () => (await log()).filter((c) => c.startsWith("updateSettings") && c.includes('"trim"')).length).toBeGreaterThan(0);

  const calls = await log();
  const patches = calls.filter((c) => c.startsWith("updateSettings"));
  // Add, the pick (target, amounts and the row), the slider, the delete, the edit of Main.
  expect(patches).toHaveLength(5);
  expect(patches[3]).toContain('"remove":[2]');
  expect(calls.filter((c) => c.startsWith("viewportSetMode") && c.includes('"node":2'))).toEqual([]);
});
