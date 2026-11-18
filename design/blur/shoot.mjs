// Renders every page of html/pages/ to a JPEG next to this file (1440 x 900 CSS px, deviceScaleFactor 2, quality 85; the sheets are full-page at 1x).
// Run:  node shoot.mjs [name ...]      (Playwright is found through NODE_PATH; set PLAYWRIGHT_BROWSERS_PATH to a folder with the installed Chromium, never run `playwright install`)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const shots = {
  "01": "01-soft", "02": "02-lens", "03": "03-motion", "04": "04-zoom", "05": "05-mask-tab", "06": "06-show-mask", "07": "07-add-node-menu", "08": "08-simple-view",
  "09": "09-respect-mask-edges", "10": "10-types", states: "states",
};
const sheets = new Set(["09", "10", "states"]);
const only = process.argv.slice(2);
const browser = await chromium.launch();
for (const [id, out] of Object.entries(shots)) {
  if (only.length && !only.includes(id)) continue;
  const sheet = sheets.has(id);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: sheet ? 1.5 : 2 });
  await page.goto("file://" + path.join(here, "html", "pages", id + ".html"));
  await page.waitForFunction(() => document.documentElement.dataset.ready === "1" || document.querySelector(".mx-sheet"));
  await page.evaluate(() => {   // the segmented control's indicator follows the checked item, as the component does
    for (const seg of document.querySelectorAll('[data-slot="segmented-control"]')) {
      const on = seg.querySelector('[data-state="on"]'); if (!on) continue;
      const r = on.getBoundingClientRect(), s = seg.getBoundingClientRect();
      seg.style.setProperty("--indicator-x", (r.left - s.left) + "px"); seg.style.setProperty("--indicator-w", r.width + "px"); seg.style.setProperty("--indicator-h", r.height + "px");
    }
  });
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(here, out + ".jpg"), type: "jpeg", quality: 85, fullPage: sheet });
  console.log("wrote", out + ".jpg");
  await page.close();
}
await browser.close();
