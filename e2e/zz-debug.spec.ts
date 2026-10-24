import { expect, test } from "@playwright/test";
import { openApp, openPhotos } from "./helpers/app";

for (const [name, openKey, wait] of [["Enter+wait", "Enter", 600], ["ArrowDown", "ArrowDown", 0], ["Enter nowait", "Enter", 0]] as const) {
  test(`debug ${name}`, async ({ page }) => {
    await openApp(page, { mockCurrent: 3, mockPicked: "2,3,4" });
    await openPhotos(page);
    await page.locator(".ym-tools__scroll").evaluate((el) => { el.scrollTop = el.scrollHeight; });
    const box = page.getByRole("combobox", { name: "Lens" });
    await box.focus();
    await page.keyboard.press(openKey);
    await expect(page.getByRole("option")).toHaveCount(7);
    if (wait) await page.waitForTimeout(wait);
    const active = async () => page.evaluate(() => { const a = document.activeElement as HTMLElement; return `${a?.getAttribute("role")}:${a?.textContent?.slice(0, 12)}`; });
    const log: string[] = [await active()];
    for (let i = 0; i < 3; i++) { await page.keyboard.press("ArrowDown"); await page.waitForTimeout(50); log.push(await active()); }
    console.log(name, log.join(" | "));
  });
}
