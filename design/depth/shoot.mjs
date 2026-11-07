// Shoots every page of design/depth/html/pages at 1440 x 900, 2x, as JPEG (the Film set's form).
// Usage: node shoot.mjs   (from design/depth; needs node_modules/playwright of the repository and the installed Chromium)
import { createRequire } from 'node:module';
import { readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
const require = createRequire('/home/user/yellowmoon/package.json');
const { chromium } = require('playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const names = { '01': '01-add-mask-menu', '02': '02-finding', '03': '03-depth-card', '04': '04-pick-depth', '04b': '04b-pick-depth-click',
  '05': '05-no-focus-point', '05b': '05b-focus-centre', '06': '06-show-depth', '07': '07-no-model', '07b': '07b-cant-be-made', '08': '08-depth-and-subject' };
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
const info = {};
for (const f of readdirSync(path.join(here, 'html/pages')).filter((f) => f.endsWith('.html')).sort()) {
  const id = f.replace('.html', '');
  const page = await ctx.newPage();
  await page.goto(pathToFileURL(path.join(here, 'html/pages', f)).href);
  await page.waitForSelector('html[data-ready="1"]', { timeout: 15000 });
  await page.waitForTimeout(400);
  info[id] = await page.evaluate(() => {
    const m = document.querySelector('[data-slot="dropdown-menu-content"]');
    return m ? { menuHeight: Math.round(m.getBoundingClientRect().height), scrolls: m.scrollHeight > m.clientHeight } : {};
  });
  await page.screenshot({ path: path.join(here, (names[id] || id) + '.jpg'), type: 'jpeg', quality: 80 });
  await page.close();
}
writeFileSync(path.join(here, 'html/measure.json'), JSON.stringify(info, null, 1));
console.log(JSON.stringify(info));
await browser.close();
