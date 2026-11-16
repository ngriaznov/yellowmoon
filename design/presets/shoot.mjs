// Renders every page of html/pages/ to a 2x JPEG next to this file (1440 x 900 CSS px, deviceScaleFactor 2, quality 85).
// Run:  node shoot.mjs [id ...]      (Playwright is found through NODE_PATH, as in this repository's agent images)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const shots = {
  "01": "01-look-tab-presets", "02": "02-empty-state", "03": "03-save-dialog", "04": "04-replace-question",
  "05": "05-tile-menu", "06": "06-missing-look-notice", "07": "07-looks-menu", "08": "08-older-export-line",
};
const only = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
for (const [id, out] of Object.entries(shots)) {
  if (only.length && !only.includes(id)) continue;
  await page.goto("file://" + path.join(here, "html", "pages", id + ".html"));
  await page.evaluate(async () => {
    for (const w of ["400", "600", "700", "800"]) await document.fonts.load(`${w} 14px Figtree`);
    await document.fonts.ready;
  });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(here, out + ".jpg"), type: "jpeg", quality: 85 });
  console.log("wrote", out + ".jpg");
}
await browser.close();
