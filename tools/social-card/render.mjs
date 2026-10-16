// Render one variant of card.html at 1280 x 640 (deviceScaleFactor 1), check it, then optimise the PNG.
// The repository's social preview (docs/images/social-card.png) is variant B over the README's Look tab screenshot;
// tools/social-card/README.md says how to make it again.
//
//   node tools/social-card/render.mjs --variant B --shot <png> --out docs/images/social-card.png
//   node tools/social-card/render.mjs --variant C --photo <jpg> --out <png>
//
// Options
//   --variant A|B|C      which layout (default A)
//   --shot <file>        the app screenshot (A, B; C when --photo is not given)
//   --photo <file>       the developed photo for C
//   --photo-pos "50% 38%"  CSS object-position of the photo in C
//   --crop x,y,w,h       C without --photo: the photo cell of the screenshot, as fractions
//   --out <png>          output PNG (required)
//   --jpeg [q]           also write <out>.jpg at quality q (default 92) to compare size and quality
//   --colors N           palette size for pngquant (default 256)
//   --no-quant           skip pngquant; only run optipng on the 24-bit PNG
//
// The run fails (non-zero exit) if Figtree did not load, an image is missing, text is closer than 64 px to an edge,
// or text overlaps the screenshot. It never falls back to a system font.

import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, writeFileSync, statSync, copyFileSync, mkdirSync, rmSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const FONT = resolve(here, '../../design/fonts/figtree-latin-wght-normal.woff2');
const WIDTH = 1280, HEIGHT = 640, MARGIN = 64;

export function parseArgs(argv) {
  const o = { variant: 'A', colors: 256, quant: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--variant') o.variant = String(next()).toUpperCase();
    else if (a === '--shot') o.shot = next();
    else if (a === '--photo') o.photo = next();
    else if (a === '--photo-pos') o.photoPos = next();
    else if (a === '--crop') o.crop = next();
    else if (a === '--out') o.out = next();
    else if (a === '--colors') o.colors = Number(next());
    else if (a === '--no-quant') o.quant = false;
    else if (a === '--jpeg') o.jpeg = argv[i + 1] && /^\d+$/.test(argv[i + 1]) ? Number(argv[++i]) : 92;
    else if (a === '--help' || a === '-h') { o.help = true; }
    else throw new Error(`unknown option ${a}`);
  }
  return o;
}

async function loadPlaywright() {
  const candidates = [process.env.YM_PLAYWRIGHT, '@playwright/test', 'playwright'].filter(Boolean);
  for (const c of candidates) {
    try { return await import(c.startsWith('/') ? pathToFileURL(c).href : c); } catch { /* try the next */ }
  }
  throw new Error('playwright not found: run npm ci, or set YM_PLAYWRIGHT to its index.mjs');
}

/** card.html with the Figtree file put in as a data URI, written to `dir`; returns its path. */
export function writeCard(dir) {
  const html = readFileSync(join(here, 'card.html'), 'utf8');
  if (html.split('__FIGTREE_WOFF2__').length !== 2) throw new Error('card.html must name __FIGTREE_WOFF2__ exactly once');
  const font = `data:font/woff2;base64,${readFileSync(FONT).toString('base64')}`;
  const page = join(dir, 'card.html');
  writeFileSync(page, html.replace('__FIGTREE_WOFF2__', () => font));
  return page;
}

function findChromium() {
  const root = process.env.PLAYWRIGHT_BROWSERS_PATH || '/opt/pw-browsers';
  if (!existsSync(root)) return undefined;
  for (const d of readdirSync(root).sort().reverse()) {
    for (const rel of ['chrome-linux/chrome', 'chrome-linux/headless_shell', 'chrome-headless-shell-linux64/chrome-headless-shell']) {
      const p = join(root, d, rel);
      if (existsSync(p)) return p;
    }
  }
  return undefined;
}

async function launch(chromium) {
  try { return await chromium.launch({ args: ['--force-color-profile=srgb', '--font-render-hinting=none'] }); }
  catch (e) {
    const exe = findChromium();
    if (!exe) throw e;
    console.error(`bundled chromium did not launch (${String(e.message).split('\n')[0]}); using ${exe}`);
    return await chromium.launch({ executablePath: exe, args: ['--force-color-profile=srgb', '--font-render-hinting=none'] });
  }
}

const toUrl = (p) => (p && !/^(file|https?|data):/.test(p) ? pathToFileURL(resolve(p)).href : p);
const size = (p) => statSync(p).size;
const kb = (n) => `${(n / 1024).toFixed(1)} KB`;

async function render(opts, rawPng, cardPage) {
  const pw = await loadPlaywright();
  const chromium = pw.chromium ?? pw.default?.chromium;
  const browser = await launch(chromium);
  try {
    const ctx = await browser.newContext({ viewport: { width: WIDTH, height: HEIGHT }, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    const pageErrors = [];
    page.on('pageerror', (e) => pageErrors.push(String(e.message ?? e)));
    const cfg = { variant: opts.variant };
    if (opts.shot) cfg.shot = toUrl(opts.shot);
    if (opts.photo) cfg.photo = toUrl(opts.photo);
    if (opts.photoPos) cfg.photoPos = opts.photoPos;
    if (opts.crop) cfg.crop = opts.crop;
    const url = pathToFileURL(cardPage).href + '?cfg=' + encodeURIComponent(JSON.stringify(cfg));
    await page.goto(url, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__cardReady === true || window.__cardError, null, { timeout: 20000 });
    const err = await page.evaluate(() => window.__cardError);
    if (err) throw new Error(err);

    // Figtree must be a loaded FontFace, and the text must be laid out with it (width differs from a fallback measure).
    const font = await page.evaluate(async () => {
      await document.fonts.ready;
      const faces = [...document.fonts].filter((f) => f.family.replace(/["']/g, '') === 'Figtree');
      const loaded = faces.length > 0 && faces.every((f) => f.status === 'loaded');
      const checks = ['800 80px Figtree', '500 30px Figtree', '400 20px Figtree'].map((s) => document.fonts.check(s));
      const el = document.querySelector('.v:not([style*="none"]) .wordmark, body[data-variant] .wordmark');
      const used = el ? getComputedStyle(el).fontFamily : '';
      return { loaded, faces: faces.length, checks, used };
    });
    if (!font.loaded || !font.checks.every(Boolean) || !/Figtree/.test(font.used)) {
      throw new Error(`Figtree did not load: ${JSON.stringify(font)}`);
    }

    // Layout guard: text and mark stay MARGIN px from every edge and never touch the screenshot frame.
    const problems = await page.evaluate(({ W, H, M }) => {
      const out = [];
      const visible = (el) => !!(el.offsetParent || el.getClientRects().length) && getComputedStyle(el).display !== 'none';
      const rect = (el) => el.getBoundingClientRect();
      const shot = [...document.querySelectorAll('[data-shot]')].filter(visible).map(rect);
      for (const el of document.querySelectorAll('[data-text]')) {
        if (!visible(el) || !el.closest('.v') || getComputedStyle(el.closest('.v')).display === 'none') continue;
        // measure the ink box of the text, not the block: use a Range for text nodes
        let r = rect(el);
        if (el.tagName !== 'svg') {
          const range = document.createRange(); range.selectNodeContents(el);
          const rr = range.getBoundingClientRect(); if (rr.width) r = rr;
        }
        const name = el.className && el.className.baseVal !== undefined ? 'mark' : (el.className || el.tagName);
        if (r.left < M - 0.5) out.push(`${name}: left ${r.left.toFixed(1)} < ${M}`);
        if (r.top < M - 0.5) out.push(`${name}: top ${r.top.toFixed(1)} < ${M}`);
        if (r.right > W - M + 0.5) out.push(`${name}: right edge ${(W - r.right).toFixed(1)} < ${M}`);
        if (r.bottom > H - M + 0.5) out.push(`${name}: bottom edge ${(H - r.bottom).toFixed(1)} < ${M}`);
        for (const s of shot) {
          if (r.left < s.right + 24 && r.right > s.left - 24 && r.top < s.bottom && r.bottom > s.top) {
            out.push(`${name}: within 24 px of the screenshot frame`);
          }
        }
      }
      return out;
    }, { W: WIDTH, H: HEIGHT, M: MARGIN });
    if (problems.length) throw new Error('layout check failed:\n  ' + problems.join('\n  '));

    if (pageErrors.length) throw new Error('page errors:\n  ' + pageErrors.join('\n  '));
    await page.screenshot({ path: rawPng, type: 'png', clip: { x: 0, y: 0, width: WIDTH, height: HEIGHT } });
  } finally {
    await browser.close();
  }
}

function run(cmd, args) { return execFileSync(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] }); }

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help) { console.log('usage: node tools/social-card/render.mjs --variant A|B|C --shot <png> [--photo <jpg>] --out <png> [--jpeg [q]]'); return; }
  if (!['A', 'B', 'C'].includes(opts.variant)) throw new Error('--variant must be A, B or C');
  if (opts.variant !== 'C' && !opts.shot) throw new Error('--shot <png> is required for variant A and B');
  if (opts.variant === 'C' && !opts.shot && !opts.photo) throw new Error('variant C needs --photo <jpg> or --shot <png>');
  for (const f of [opts.shot, opts.photo]) if (f && !/^(file|https?|data):/.test(f) && !existsSync(f)) throw new Error(`not found: ${f}`);
  if (!opts.out) throw new Error('--out <png> is required');
  const out = resolve(opts.out);
  mkdirSync(dirname(out), { recursive: true });

  const tmp = mkdtempSync(join(tmpdir(), 'ym-social-card-'));
  const raw = join(tmp, 'raw.png'), quant = join(tmp, 'quant.png');
  try {
    await render(opts, raw, writeCard(tmp));
    const rawSize = size(raw);
    let src = raw;
    if (opts.quant) {
      // 256-colour palette with Floyd-Steinberg dithering (pngquant's default), slowest speed for the best palette.
      run('pngquant', ['--force', '--speed', '1', '--quality', '85-100', '--floyd=1', '--output', quant, String(opts.colors), raw]);
      src = quant;
    }
    copyFileSync(src, out);
    run('optipng', ['-quiet', '-o5', '-strip', 'all', out]);
    const finalSize = size(out);
    console.log(`${opts.variant}: raw ${kb(rawSize)}${opts.quant ? ` -> pngquant ${kb(size(quant))}` : ''} -> optipng ${kb(finalSize)}  ${out}`);
    if (finalSize >= 1024 * 1024) console.error('warning: the PNG is over 1 MB, GitHub will refuse it');
    if (opts.jpeg) {
      const jpg = out.replace(/\.png$/i, '') + '.jpg';
      run('convert', [raw, '-strip', '-sampling-factor', '4:2:0', '-quality', String(opts.jpeg), '-interlace', 'none', jpg]);
      console.log(`${opts.variant}: jpeg q${opts.jpeg} ${kb(size(jpg))}  ${jpg}`);
    }
    if (process.env.YM_KEEP_RAW) copyFileSync(raw, out.replace(/\.png$/i, '') + '.raw.png');
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((e) => { console.error(String(e.message ?? e)); process.exit(1); });
}
