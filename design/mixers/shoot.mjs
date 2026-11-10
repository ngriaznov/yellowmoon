// Renders every page of html/pages/ to a 2x JPEG next to this file (1440 x 900 CSS px, deviceScaleFactor 2, quality 85).
// Run:  node shoot.mjs [name ...]      (Playwright is found through NODE_PATH, as in this repository's agent images)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const shots = {
  "01": "01-add-node-menu", "02": "02-parallel", "03": "03-layer-blend", "04": "04-layer-mode-menu",
  "05": "05-splitter-and-combiner", "06": "06-combiner", "07": "07-drag-to-plus", states: "states",
};
const only = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
for (const [id, out] of Object.entries(shots)) {
  if (only.length && !only.includes(id)) continue;
  await page.goto("file://" + path.join(here, "html", "pages", id + ".html"));
  await page.waitForFunction(() => document.documentElement.dataset.ready === "1" || document.querySelector(".mx-sheet"));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(here, out + ".jpg"), type: "jpeg", quality: 85, fullPage: id === "states" });
  console.log("wrote", out + ".jpg");
}
await browser.close();
