// Writes the pages of design/presets/html/pages/ (task P-01).
//
// The pages are the DOM of the app itself, taken from the BUILT app on its mock API (the flat-shape photos of the mock, no photo of the owner's):
// the Look tab of the Photos screen (the Tone card taken out, the Presets card added after the Look card), the dialogs, the tile menu and the notice added to it by hand with the app's own classes
// (`presets.css` has the few rules the app does not have: the checkbox, the macOS menu bar). The Presets card, the Save dialog and the menu are
// MOCK-UPS, not product code: nothing here is in `src/`.
//
// Run:  node capture.mjs <dist folder of a built app> [state ...]       (Playwright is found through NODE_PATH, as in this repository's agent images)
//       e.g. node capture.mjs /path/to/yellowmoon/dist           (npm run build makes dist/)
// The stylesheets (html/app.css, html/graph.css) and the font were copied from that dist/ once: `cp dist/assets/glass-c2-pages-*.css html/app.css`,
// with the two url(/assets/figtree-…) rewritten to assets/fonts/figtree-….woff2. Pictures are made from the pages by shoot.mjs.
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import path from "node:path";
const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); } catch { ({ chromium } = require(path.join(process.env.NODE_PATH ?? "", "playwright"))); }
const here = path.dirname(fileURLToPath(import.meta.url));
const DIST = path.resolve(process.argv[2] ?? "dist");
const only = process.argv.slice(3);
const MIME = { ".js": "text/javascript", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml", ".woff2": "font/woff2" };

// ---------------------------------------------------------------------------------------------------------------------------------- the data
const PRESETS = [
  { name: "Warm evening", groups: "Look, Light, Colour", filter: "sepia(.35) saturate(1.25) hue-rotate(-8deg) brightness(.97)" },
  { name: "Cool morning", groups: "Light, Colour", filter: "hue-rotate(40deg) saturate(.8) brightness(1.1)" },
  { name: "Soft portrait", groups: "Look, Light, Detail", filter: "brightness(1.12) contrast(.8) saturate(1.3) hue-rotate(-20deg)" },
  { name: "Street grit", groups: "Look, Light, Curve, Detail", filter: "contrast(1.5) saturate(.3) brightness(.85)" },
  { name: "Faded film", groups: "Look, Curve", filter: "sepia(.45) contrast(.75) saturate(.7) brightness(1.1)" },
  { name: "Night blue", groups: "Light, Colour, Curve", filter: "hue-rotate(28deg) brightness(.72) saturate(1.1)" },
];
const GROUP_ROWS = [
  ["Look", "The look and how strongly it is applied.", true],
  ["Lens", "The lens and its amounts.", false],
  ["Light", "Exposure, contrast, brightness, highlights, shadows, white and black.", true],
  ["Colour", "White balance, colour balance, saturation and colour warp.", true],
  ["Detail", "Sharpening, noise reduction and output sharpening.", false],
  ["Curve", "The luma, red, green and blue curves.", false],
  ["RAW", "Decode options and the range.", false],
  ["Nodes", "The other nodes, their links and masks.", false],
];

// ----------------------------------------------------------------------------------------------------------------------------------- the app
async function openLookTab(browser, query = "") {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  await page.route("http://app.test/**", (route) => {
    const u = new URL(route.request().url());
    const p = path.join(DIST, u.pathname === "/" ? "index.html" : u.pathname);
    if (!fs.existsSync(p)) return route.fulfill({ status: 404, body: "" });
    route.fulfill({ status: 200, contentType: MIME[path.extname(p)] ?? "application/octet-stream", body: fs.readFileSync(p) });
  });
  await page.goto("http://app.test/?mockSpeed=Infinity&mockCurrent=3&mockPicked=2,3,4" + query);
  await page.evaluate(async () => {
    for (const w of ["400", "600", "700", "800"]) await document.fonts.load(`${w} 14px Figtree`);
    await document.fonts.ready;
  });
  await page.getByRole("button", { name: "Open files…" }).click();
  await page.getByRole("heading", { level: 2, name: /photos? ready/ }).waitFor();
  await page.getByRole("button", { name: /Open photos/ }).click();
  await page.getByRole("tab", { name: "Look" }).waitFor();
  await page.getByRole("region", { name: "Automatic" }).waitFor();
  await page.waitForFunction(() => document.querySelectorAll(".ym-look-tile img").length >= 2);
  await page.waitForTimeout(1200);
  return page;
}

/** Everything the in-page code needs; it is evaluated in the page. */
const IN_PAGE = {
  /** Takes the Tone card out and puts the Presets card after the Look card (the owner, 2026-11-16: docs/30 §9.10), then scrolls the panel to its end. `count` presets exist; `shown` tiles are drawn (0 draws the empty state). */
  presetsCard({ presets, count, shown, ringed, scroll }) {
    const look = [...document.querySelectorAll("section.ym-card")].find((s) => s.querySelector(".ym-card__title")?.textContent === "Look");
    const reset = look.querySelector(".ym-card__actions button");
    const field = look.querySelector(".ym-field");
    const tileSrc = [...document.querySelectorAll(".ym-look-tile")];
    const mk = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };
    const card = mk(`<section class="ym-card" aria-labelledby="pr-title" data-presets="1"><div class="ym-card__header"><h3 id="pr-title" class="ym-card__title">Presets</h3><span class="ym-card__actions"></span></div></section>`);
    const actions = card.querySelector(".ym-card__actions");
    const linkButton = (text) => { const b = reset.cloneNode(true); b.textContent = text; return b; };
    if (shown > 0) {
      actions.append(mk(`<span class="ym-look-group__count pr-count">${count}</span>`), linkButton("Save…"));
      const f = field.cloneNode(true);
      f.querySelector("[role=radiogroup]").setAttribute("aria-label", "Apply preset to");
      card.append(f);
      const body = mk(`<div class="ym-look-group__body"><div class="ym-look-grid" role="group" aria-label="Presets"></div></div>`);
      const grid = body.querySelector(".ym-look-grid");
      presets.slice(0, shown).forEach((p, i) => {
        const src = tileSrc[i % 2];
        const t = src.cloneNode(true);
        t.removeAttribute("aria-label");
        t.setAttribute("aria-label", `${p.name} preset`);
        t.title = p.groups;
        t.setAttribute("aria-pressed", i === ringed ? "true" : "false");
        t.querySelectorAll(".ym-look-tile__badge, .ym-look-tile__default").forEach((n) => n.remove());
        t.querySelector(".ym-look-tile__name").textContent = p.name;
        t.querySelector("img").style.filter = p.filter;
        grid.append(t);
      });
      card.append(body);
      if (count > shown) {
        const more = mk(`<div class="pr-more"></div>`);
        more.append(linkButton(`Show all ${count}`));
        more.firstChild.setAttribute("aria-expanded", "false");
        card.append(more);
      }
    } else {
      card.append(mk(`<div class="pr-empty"><p class="pr-empty__text">Set a photo the way you like it, then save it as a preset.</p></div>`));
      const b = document.querySelector(".ym-foot, .ym-tools__foot")?.querySelector("button");
      const save = b ? b.cloneNode(true) : document.createElement("button");
      save.textContent = "Save preset…";
      save.removeAttribute("disabled");
      save.removeAttribute("aria-disabled");
      card.querySelector(".pr-empty").append(save);
    }
    [...document.querySelectorAll("section.ym-card")].filter((c) => c.querySelector(".ym-card__title")?.textContent === "Tone").forEach((c) => c.remove());
    look.after(card);
    document.querySelector(".ym-tools__scroll").scrollTop = 1e6;
  },
  /** The look chip and the photo, as they are with Warm evening applied. */
  warmPhoto() {
    const chip = [...document.querySelectorAll("*")].find((e) => e.children.length === 0 && /^NEUTRAL · 100%$/i.test(e.textContent.trim()));
    if (chip) chip.textContent = chip.textContent.replace(/neutral/i, "Film warm");
    const img = document.querySelector(".ym-photo-cell__image");
    if (img) img.style.filter = "sepia(.3) saturate(1.2) hue-rotate(-8deg)";
  },
  /** Opens the app's own dialog (Preferences) and turns it into `spec`: a title, a body and two footer buttons. */
  async dialog({ title, width, bodyHtml, cancel, confirm }) {
    const d = document.querySelector("[role=dialog]");
    d.style.width = width + "px";
    d.querySelector("[data-slot=dialog-title]").textContent = title;
    const body = d.querySelector("[data-slot=dialog-body]");
    body.innerHTML = bodyHtml;
    const footer = d.querySelector("[data-slot=dialog-footer]");
    const primary = footer.querySelector("[data-variant=primary]");
    const link = footer.querySelector(".ym-dialog__footer-start");
    primary.textContent = confirm;
    const glass = document.querySelector(".ym-tools__foot button, [aria-label='Copy and paste settings'] button");
    const cancelBtn = glass.cloneNode(true);
    cancelBtn.textContent = cancel;
    cancelBtn.removeAttribute("disabled");
    link.replaceWith(cancelBtn);
    footer.insertBefore(cancelBtn, footer.firstElementChild);
    footer.style.justifyContent = "flex-end";
    footer.style.gap = "10px";
    return d;
  },
};

// ------------------------------------------------------------------------------------------------------------------------------ the states
const GROUP_BODY = (rows, nameValue, noteHtml = "") => `
  <div class="pr-name"><label class="pr-label" for="pr-name">Name</label>
    <input id="pr-name" data-slot="input" class="pr-input" type="text" placeholder="Name" value="${nameValue}" maxlength="64" aria-label="Name"></div>
  <div class="pr-groups" role="group" aria-label="Save these">${rows
    .map(
      ([label, line, on]) => `<div class="pr-group" role="checkbox" aria-checked="${on}" aria-labelledby="g-${label}" tabindex="0">
        <span class="pr-check" data-checked="${on}" aria-hidden="true"><svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span>
        <span class="pr-group__text"><span id="g-${label}" class="pr-group__label">${label}</span><span class="pr-group__line">${line}</span></span></div>`,
    )
    .join("")}</div>
  <p class="pr-note">Auto stays Auto, so each photo keeps its own exposure and white balance.</p>${noteHtml}`;

const STATES = {
  "01": { title: "Look tab with the Presets card", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(IN_PAGE.warmPhoto);
  } },
  "02": { title: "Presets card, empty", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: [], count: 0, shown: 0, ringed: -1, scroll: 0 });
  } },
  "03": { title: "Save preset dialog", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: -1, scroll: 0 });
    await page.keyboard.press("Meta+,");
    await page.waitForSelector("[role=dialog]");
    await page.waitForTimeout(500);
    await page.evaluate(IN_PAGE.dialog, { title: "Save preset", width: 480, bodyHtml: GROUP_BODY(GROUP_ROWS, "Warm evening"), cancel: "Cancel", confirm: "Save" });
  } },
  "04": { title: "Replace Warm evening?", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: -1, scroll: 0 });
    await page.keyboard.press("Meta+,");
    await page.waitForSelector("[role=dialog]");
    await page.waitForTimeout(500);
    await page.evaluate(IN_PAGE.dialog, {
      title: "Replace Warm evening?", width: 480,
      bodyHtml: `<p class="pr-question">A preset with this name is already saved. Replacing it can't be undone.</p>`, cancel: "Cancel", confirm: "Replace",
    });
  } },
  "05": { title: "A tile's menu", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(IN_PAGE.warmPhoto);
    await page.locator('button[aria-label^="Show DSC"]').first().click({ button: "right" });
    await page.waitForSelector("[role=menu]");
    await page.waitForTimeout(400);
    await page.evaluate(() => {
      const menu = document.querySelector("[role=menu]");
      const item = menu.querySelector("[role=menuitem]");
      const names = [["Apply to Picked Photos", false], ["Update from This Photo", false], ["Rename…", false], ["Export…", false], ["Show in Finder", false], ["Delete", false]];
      const sep = () => { const s = document.createElement("div"); s.className = "pr-menu-sep"; s.setAttribute("role", "separator"); return s; };
      menu.setAttribute("aria-label", "Cool morning actions");
      menu.replaceChildren(
        ...names.map(([n], i) => {
          const c = item.cloneNode(true);
          c.textContent = n;
          if (i === 5) c.setAttribute("data-variant", "destructive");
          return c;
        }),
      );
      menu.insertBefore(sep(), menu.children[2]);
      menu.insertBefore(sep(), menu.children[6]);
      const tile = document.querySelectorAll('[data-presets] .ym-look-tile')[1];
      const r = tile.getBoundingClientRect();
      const wrap = menu.parentElement;
      wrap.dataset.prPos = `${Math.round(r.left + r.width * 0.5)}px, ${Math.round(r.top + r.height * 0.1)}px`;
      document.querySelectorAll("[data-state=open]").forEach((n) => { if (n.matches("button")) n.setAttribute("data-state", "closed"); });
      tile.classList.add("pr-menu-open");
    });
  } },
  "06": { title: "A missing look", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: -1, scroll: 0 });
    await page.getByRole("button", { name: "Copy settings" }).click();
    await page.waitForTimeout(500);
    const ok = await page.evaluate(() => {
      const bar = document.querySelector("[role=status]:not(.sr-only):not([aria-live=polite])") ?? [...document.querySelectorAll("[role=status],[role=alert]")].find((e) => /Copied/.test(e.textContent));
      if (!bar) return false;
      const t = [...bar.querySelectorAll("*")].find((e) => e.children.length === 0 && /Copied/.test(e.textContent)) ?? bar;
      t.textContent = "Warm evening uses the look Film warm, which isn't imported. Everything else was applied.";
      return true;
    });
    if (!ok) throw new Error("the notice bar was not found");
    await page.waitForTimeout(300);
  } },
  "07": { title: "The Looks menu", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(IN_PAGE.warmPhoto);
    await page.evaluate(MENU_BAR);
  } },
  "08": { title: "The older-export line", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: -1, scroll: 0 });
    await page.keyboard.press("Meta+,");
    await page.waitForSelector("[role=dialog]");
    await page.waitForTimeout(500);
    await page.evaluate(IN_PAGE.dialog, {
      title: "Save preset", width: 480, cancel: "Cancel", confirm: "Save",
      bodyHtml: GROUP_BODY(
        GROUP_ROWS.map(([l, t, _on]) => [l, t, l === "Look"]),
        "",
        `<p class="pr-note pr-note--older">This photo was exported by an older Yellowmoon, so its automatic values are saved as fixed numbers. Untick Light, Colour and Detail to leave them out.</p>`,
      ),
    });
  } },

  "09": { title: "The File menu", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(MAC, { menu: "File", left: 140, width: 290, rows: [
      ["Open Files…", "⌘O"], ["Open Folder…", "⇧⌘O"], ["-"],
      ["Save", "⌘S", { fresh: true, hover: true }], ["Save As…", "⇧⌘S", { fresh: true }], ["-"],
      ["Export…", "⌘E"], ["-"], ["Close Session", "⌘W"],
    ] });
  } },
  "10": { title: "The Save panel", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(MAC, { panel: `<div class="pr-dim"></div><div class="pr-sheet" role="dialog" aria-label="Save">
      <div class="pr-sheet__row"><label>Save As:</label><span class="pr-sheet__field">Lisbon.ymalbum</span></div>
      <div class="pr-sheet__row"><label>Where:</label><span class="pr-sheet__popup">Lisbon <i>⌃</i></span></div>
      <div class="pr-sheet__list"><div class="pr-sheet__item"><b>DSC04201.ARW</b></div><div class="pr-sheet__item"><b>DSC04202.ARW</b></div><div class="pr-sheet__item"><b>DSC04203.ARW</b></div></div>
      <div class="pr-sheet__buttons"><span class="pr-btn">New Folder</span><span class="pr-sp"></span><span class="pr-btn">Cancel</span><span class="pr-btn pr-btn--default">Save</span></div></div>` });
  } },
  "11": { title: "An album open and edited", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(() => {
      const sub = document.querySelector(".ym-brand-sub");
      sub.textContent = sub.textContent.replace(/^[^·]*· /, "") + " · Edited";
    });
    await page.evaluate(MAC, { menu: "Window", left: 392, width: 260, lights: true, shift: true, edited: true, rows: [
      ["Minimize", "⌘M"], ["Zoom", ""], ["-"], ["Lisbon — Edited", "", { check: true }],
    ] });
  } },
  "12": { title: "Save changes to the album?", async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: 0, scroll: 0 });
    await page.evaluate(() => {
      const sub = document.querySelector(".ym-brand-sub");
      sub.textContent = sub.textContent.replace(/^[^·]*· /, "") + " · Edited";
    });
    await page.evaluate(MAC, { bar: false, lights: true, edited: true, panel: `<div class="pr-dim"></div><div class="pr-alert" role="alertdialog" aria-label="Save changes to the album Lisbon?">
      <span class="pr-alert__icon" aria-hidden="true"></span>
      <b class="pr-alert__title">Save changes to the album Lisbon?</b>
      <p class="pr-alert__text">Your changes will be lost if you don't save them.</p>
      <span class="pr-btn pr-btn--default pr-btn--wide">Save</span><span class="pr-btn pr-btn--wide">Cancel</span><span class="pr-btn pr-btn--wide">Don't Save</span></div>` });
  } },
  "13": { title: "An album with a missing photo", async query() { return "&mockVanish=DSC04206"; }, async run(page) {
    await page.evaluate(IN_PAGE.presetsCard, { presets: PRESETS, count: 14, shown: 6, ringed: -1, scroll: 0 });
    await page.locator('button[aria-label^="Show DSC04206"]').first().click();
    await page.waitForTimeout(1500);
    await page.evaluate(() => {
      const hit = [...document.querySelectorAll("*")].filter((e) => e.children.length === 0 && /\S/.test(e.textContent));
      const reason = hit.find((e) => /gone|moved|find this file|can't find/i.test(e.textContent));
      if (reason) reason.textContent = "Not found: /Users/anna/Pictures/Lisbon/DSC04206.ARW";
      window.__reason = reason ? "found" : "missing";
    });
    console.log("  reason text:", await page.evaluate(() => window.__reason));
  } },
};


/** The macOS menu bar with one menu open, and (for the album states) pieces of macOS's own windows: the Save panel, an alert, the traffic lights. */
function MAC(spec) {
  const chk = (on) => `<span class="mm-chk">${on ? "✓" : ""}</span>`;
  const row = ([label, key, opts = {}]) =>
    label === "-" ? '<div class="mm-sep"></div>' : `<div class="mm-item${opts.hover ? " mm-item--hover" : ""}${opts.fresh ? " mm-item--new" : ""}">${chk(opts.check)}<span class="mm-label">${label}</span><span class="mm-key">${key ?? ""}</span></div>`;
  const root = document.createElement("div");
  root.className = "pr-macos";
  const bar = ["File", "Edit", "View", "Photo", "Looks", "Window", "Help"].map((m) => `<span${m === spec.menu ? ' class="mm-bar--on"' : ""}>${m}</span>`).join("");
  let html = spec.bar === false ? "" : `<div class="mm-bar"><span class="mm-apple"></span><b>Yellowmoon</b>${bar}</div>`;
  if (spec.shift) document.querySelector("#root").style.cssText += "position:relative;top:25px;";
  if (spec.menu) html += `<div class="mm-menu" style="left:${spec.left}px;width:${spec.width ?? 280}px" role="menu" aria-label="${spec.menu}">${spec.rows.map(row).join("")}</div>`;
  if (spec.lights) html += `<div class="pr-lights" style="top:${spec.shift ? 47 : 22}px"><i class="r">${spec.edited ? "<u></u>" : ""}</i><i class="y"></i><i class="g"></i></div>`;
  if (spec.panel) html += spec.panel;
  root.innerHTML = html;
  document.body.append(root);
}

/** The macOS menu bar and the open Looks menu with its Apply Preset submenu: macOS draws them, not the app; this is their look. */
function MENU_BAR() {
  const chk = (on) => `<span class="mm-chk">${on ? "✓" : ""}</span>`;
  const item = (label, key = "", opts = {}) =>
    `<div class="mm-item${opts.hover ? " mm-item--hover" : ""}${opts.off ? " mm-item--off" : ""}${opts.fresh ? " mm-item--new" : ""}">${chk(opts.check)}<span class="mm-label">${label}</span><span class="mm-key">${key}</span></div>`;
  const sep = '<div class="mm-sep"></div>';
  const looks = [1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => item(`Apply Look ${n}`)).join("");
  const sub = ["Warm evening", "Cool morning", "Soft portrait", "Street grit", "Faded film", "Night blue"].map((n, i) => item(n, "", { check: i === 0 })).join("");
  const root = document.createElement("div");
  root.className = "pr-macos";
  root.innerHTML = `
    <div class="mm-bar"><span class="mm-apple"></span><b>Yellowmoon</b><span>File</span><span>Edit</span><span>View</span><span>Photo</span><span class="mm-bar--on">Looks</span><span>Window</span><span>Help</span></div>
    <div class="mm-menu" role="menu" aria-label="Looks">
      ${item("Import ARRI Looks…")}${item("Import LUTs and look files…")}${item("Make Current Look Default")}${sep}${looks}${sep}
      ${item("Save Preset…", "⌥⌘S", { fresh: true })}${item("Import Presets…", "", { fresh: true })}
      <div class="mm-item mm-item--hover mm-item--new">${chk(false)}<span class="mm-label">Apply Preset</span><span class="mm-key">▸</span></div>
      ${item("New Preset from Exported Photo…", "", { fresh: true })}${sep}${item("Show Presets Folder", "", { fresh: true })}${item("Show Looks Folder")}
    </div>
    <div class="mm-menu mm-menu--sub" role="menu" aria-label="Apply Preset">${sub}</div>`;
  document.body.append(root);
}

// ------------------------------------------------------------------------------------------------------------------------------ the page file
/** Turns the live page into a file that stands alone: blob pictures become data, the app's scripts go, the links point at the folder's sheets. */
async function savePage(page, file, title) {
  await page.evaluate(async () => {
    for (const img of document.querySelectorAll("img[src^='blob:']")) {
      const blob = await (await fetch(img.src)).blob();
      img.src = await new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(blob); });
    }
  });
  let html = await page.evaluate(() => {
    document.querySelectorAll("script, link[rel=modulepreload], link[rel=icon]").forEach((n) => n.remove());
    document.querySelectorAll("link[rel=stylesheet]").forEach((n) => n.remove());
    // Radix keeps moving a menu to where the right click was; the mock-up wants it beside its tile, so the place is set at the last moment.
    document.querySelectorAll("[data-pr-pos]").forEach((w) => {
      w.style.setProperty("transform", `translate(${w.dataset.prPos})`, "important");
      document.querySelectorAll("button[data-state=open]").forEach((b) => b.setAttribute("data-state", "closed"));
      document.querySelectorAll("[data-menu=open]").forEach((b) => b.removeAttribute("data-menu"));
    });
    document.title = "";
    document.body.removeAttribute("style");
    document.body.removeAttribute("data-scroll-locked");
    document.querySelectorAll("[data-radix-focus-guard]").forEach((n) => n.remove());
    document.querySelectorAll("[data-aria-hidden]").forEach((n) => { n.removeAttribute("data-aria-hidden"); n.removeAttribute("aria-hidden"); });
    return "<!doctype html>\n" + document.documentElement.outerHTML;
  });
  html = html.replace("<title></title>", `<title>${title}</title>`);
  const links = `<link href="../app.css" rel="stylesheet"><link href="../graph.css" rel="stylesheet"><link href="../presets.css" rel="stylesheet">`;
  html = html.replace(/<title>[^<]*<\/title>/, (m) => `${m}${links}`);
  if (!html.includes(links)) html = html.replace("</head>", `${links}</head>`);
  // A saved page has no scroll position: the tool panel is scrolled to its end when the page loads, as the picture shows it.
  html = html.replace("</body>", `<script>addEventListener("load", function () { var s = document.querySelector(".ym-tools__scroll"); if (s) s.scrollTop = 1e6; });</script></body>`);
  fs.writeFileSync(file, html);
}

const browser = await chromium.launch();
fs.mkdirSync(path.join(here, "html", "pages"), { recursive: true });
for (const [id, state] of Object.entries(STATES)) {
  if (only.length && !only.includes(id)) continue;
  const page = await openLookTab(browser, state.query ? await state.query() : "");
  await state.run(page);
  await page.waitForTimeout(400);
  await savePage(page, path.join(here, "html", "pages", id + ".html"), `Presets · ${state.title}`);
  console.log("wrote", id, state.title);
  await page.close();
}
await browser.close();
