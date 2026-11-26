// Renders every page of html/pages/ to a JPEG next to this file (1440 x 900 CSS px, deviceScaleFactor 2, quality 85; the sheets are full-page at 1.5x).
// Run:  node shoot.mjs [name ...]      (Playwright is found through NODE_PATH; set PLAYWRIGHT_BROWSERS_PATH to a folder with the installed Chromium, never run `playwright install`)
// Also prints the console errors of each page and any text that overflows its box at 1440 x 900 (the clipping check).
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const shots = {
  "01": "01-empty", "02": "02-choose-file", "02b": "02b-choose-from-strip", "03": "03-fitting", "04": "04-fitted", "05": "05-hue-tab", "06": "06-elements-tab",
  "06b": "06b-elements-sheet", "06c": "06c-element-mask", "07": "07-look-changed", "08": "08-edited-by-hand", "09": "09-add-node-menu", "10": "10-simple-view",
  "11": "11-status-and-errors", "12": "12-before-after", states: "states",
};
const sheets = new Set(["06b", "11", "12", "states"]);
const only = process.argv.slice(2);
const opts = process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {};
const browser = await chromium.launch(opts);
for (const [id, out] of Object.entries(shots)) {
  if (only.length && !only.includes(id)) continue;
  const sheet = sheets.has(id);
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: sheet ? 1.5 : 2 });
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto("file://" + path.join(here, "html", "pages", id + ".html"));
  await page.waitForFunction(() => document.documentElement.dataset.ready === "1" || document.querySelector(".mx-sheet") || document.readyState === "complete");
  await page.evaluate(() => {   // the segmented control's and the tab row's indicators follow the selected item, as the components do
    for (const seg of document.querySelectorAll('[data-slot="segmented-control"]')) {
      const on = seg.querySelector('[data-state="on"]'); if (!on) continue;
      const r = on.getBoundingClientRect(), s = seg.getBoundingClientRect();
      seg.style.setProperty("--indicator-x", (r.left - s.left) + "px"); seg.style.setProperty("--indicator-w", r.width + "px"); seg.style.setProperty("--indicator-h", r.height + "px");
    }
    for (const tl of document.querySelectorAll('[data-slot="tabs-list"]')) {
      const on = tl.querySelector('[data-state="active"]'); if (!on) continue;
      const r = on.getBoundingClientRect(), s = tl.getBoundingClientRect();
      tl.style.setProperty("--indicator-x", (r.left - s.left) + "px"); tl.style.setProperty("--indicator-w", r.width + "px");
    }
  });
  await page.waitForTimeout(300);
  const clipped = await page.evaluate(() => {
    const out = [];
    for (const el of document.querySelectorAll("button, h2, h3, p, span, label, a, td, th, figcaption")) {
      if (el.closest("svg") || el.closest(".sr-only") || el.classList.contains("sr-only")) continue;
      const cs = getComputedStyle(el);
      if (cs.display === "none" || cs.visibility === "hidden") continue;
      if (el.children.length === 0 && el.textContent.trim() && (el.scrollWidth > el.clientWidth + 1) && cs.overflow !== "visible") out.push(el.tagName + ":" + el.textContent.trim().slice(0, 40));
      if (el.children.length === 0 && el.textContent.trim() && el.scrollWidth > el.clientWidth + 1 && cs.textOverflow === "ellipsis") out.push("ellipsis:" + el.textContent.trim().slice(0, 40));
    }
    return out;
  });
  await page.screenshot({ path: path.join(here, out + ".jpg"), type: "jpeg", quality: 85, fullPage: sheet });
  console.log("wrote", out + ".jpg", errors.length ? "CONSOLE ERRORS: " + errors.join(" | ") : "", clipped.length ? "CLIPPED: " + clipped.join(" | ") : "");
  await page.close();
}
await browser.close();
