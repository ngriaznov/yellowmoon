// Renders html/assets/scene-portrait.svg (drawn by ../skin/html/faces.py) to html/assets/portrait-before.png at 1500 x 1000, for compress-demo.py.
// Run:  PLAYWRIGHT_BROWSERS_PATH=<scratch> node render-scene.mjs
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const svg = path.join(here, "..", "skin", "html", "assets", "scene-portrait.svg");
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 1 });
await page.goto("file://" + svg);
await page.screenshot({ path: path.join(here, "html", "assets", "portrait-before.png"), type: "png" });
await browser.close();
console.log("wrote html/assets/portrait-before.png");
