// Runs INSIDE the real webview of the real Tauri shell (see screenshots.mjs, which serves it with the built frontend).
// Plain browser JavaScript, no imports; the helpers come from e2e/real/kit.js.
//
// It opens the photos the way a person does (a drop on the window), waits until everything on screen is final, and asks the runner for a
// picture of the screen in each state: the Start screen with the photos ready, the Look, Light, Colour and Detail tabs, and the Export
// panel. A state is never captured while a frame, a thumbnail, a look tile or the histogram is still on its way.
(function () {
  "use strict";
  const config = window.__SMOKE__;
  const { invoke, sleep, send, waitFor, q, qa, loaded, textOf, key, frame, emit, dropPaths } = window.__YM_KIT__;

  /** The tool tabs by the name of a state: the key that picks the tab (⌘1 to ⌘4) and the words of its title block. */
  const TABS = { look: { key: "1", title: "Look" }, light: { key: "2", title: "Light" }, colour: { key: "3", title: "Colour" }, detail: { key: "4", title: "Detail" } };
  const note = (text) => send("note", { text });
  const device = (n) => Math.round(n * (window.devicePixelRatio || 1));
  const rectOf = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: device(r.left), y: device(r.top), width: device(r.width), height: device(r.height) };
  };

  /** The part of an element that can be seen: its box cut by every ancestor that clips (a scrolling list), in device pixels, or null. */
  const visibleRect = (el) => {
    let { left, top, right, bottom } = el.getBoundingClientRect();
    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      if (style.overflowX === "visible" && style.overflowY === "visible") continue;
      const r = parent.getBoundingClientRect();
      left = Math.max(left, r.left);
      top = Math.max(top, r.top);
      right = Math.min(right, r.right);
      bottom = Math.min(bottom, r.bottom);
    }
    left = Math.max(left, 0);
    top = Math.max(top, 0);
    right = Math.min(right, window.innerWidth);
    bottom = Math.min(bottom, window.innerHeight);
    return right - left > 1 && bottom - top > 1 ? { x: device(left), y: device(top), width: device(right) - device(left), height: device(bottom) - device(top) } : null;
  };
  const rectsOf = (selector) => qa(selector).map(visibleRect).filter(Boolean);

  // --- What is on screen ----------------------------------------------------------------------------------------------------------

  /** The photo the filmstrip says is open: its id, or null. */
  const currentId = () => {
    const current = q(".ym-filmstrip__thumb[aria-current=true]");
    return current ? Number(current.getAttribute("data-photo-id")) : null;
  };
  const activeTab = () => {
    const tab = q("[role=tab][aria-selected=true]");
    return tab ? tab.textContent.trim().toLowerCase() : "";
  };

  /**
   * Looks at the screen and says whether it is final (`why` lists what is still missing) and what it shows (`signature`: the same string
   * twice in a row, a moment apart, means nothing is moving). Final means: on the Photos screen, a frame of the cell's size is drawn, no
   * note ("Still developing.") covers the photo, the header has the render time, every thumbnail of the filmstrip is drawn and, on the
   * Look tab, every tile shows the open photo through its look (the address of a tile names the photo it was made from), and on the
   * Light tab the histogram has its curves.
   */
  function look() {
    const why = [];
    const screen = (q("[data-screen]") || { getAttribute: () => "" }).getAttribute("data-screen");
    if (screen === "start") {
      const tags = qa("[data-screen=start] .ym-tag").map((t) => t.textContent.trim());
      if (!/^\d+ photos? ready$/.test(textOf("[data-screen=start] h2"))) why.push("the Start screen is still developing: " + textOf("[data-screen=start] h2"));
      if (tags.some((t) => !/^(ready|skipped)$/i.test(t))) why.push("a file is not Ready yet");
    } else if (screen === "photos") {
      const cell = q("[data-viewport]");
      const f = frame();
      if (!cell) why.push("no photo cell");
      else if (!f || !loaded(f) || f.naturalWidth <= 16) why.push("no frame yet");
      else {
        const box = cell.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        if (Math.abs(f.naturalWidth - Math.round(box.width * dpr)) > 1 || Math.abs(f.naturalHeight - Math.round(box.height * dpr)) > 1) why.push(`the frame is ${f.naturalWidth}x${f.naturalHeight}, not the cell's size`);
      }
      if (q(".ym-photo-note")) why.push("a note covers the photo: " + textOf(".ym-photo-note"));
      if (/Before · camera default/.test(textOf("[data-viewport]"))) why.push("the photo is shown in Before mode");
      // (The Export panel's header says "Classical · Display P3": the render time is the photo view's.)
      if (!q("aside[aria-label=Export]") && !/· \d+ ms ·/.test(textOf(".ym-status"))) why.push("no render time in the header yet");
      const open = currentId();
      for (const thumb of qa(".ym-filmstrip__thumb")) {
        if (thumb.getAttribute("data-status") === "failed") continue;
        const img = q("img", thumb);
        if (!img) why.push("a thumbnail is missing");
        else if (!loaded(img)) why.push("a thumbnail is loading");
      }
      if (q("aside[aria-label=Export]")) {
        if (!/≈/.test(textOf("aside[aria-label=Export] .ym-tools__footer button"))) why.push("the export estimate is not there");
        if (!q("aside[aria-label=Export] code")) why.push("the command is not there");
      } else if (activeTab() === "look") {
        const tiles = qa(".ym-look-tile");
        if (tiles.length === 0) why.push("no look tiles");
        for (const tile of tiles) {
          const img = q("img", tile);
          if (!img || !loaded(img)) why.push("a look tile is loading");
          else {
            const made = /look-(\d+)-/.exec(img.src);
            if (made && open !== null && Number(made[1]) !== open) why.push("a look tile still shows another photo");
          }
        }
      } else if (activeTab() === "light") {
        if (!qa(".ym-histogram svg path").some((p) => (p.getAttribute("d") || "").length > 10)) why.push("the histogram is empty");
      }
    } else why.push("unexpected screen: " + screen);
    return { ready: why.length === 0, why: why.join("; "), signature: document.body.innerHTML + "|" + (frame() ? frame().src : "") };
  }

  /** Waits until the screen is final and has not changed for `config.stableMs`. */
  async function settle(label) {
    const started = Date.now();
    let last = null;
    let since = 0;
    for (;;) {
      const now = look();
      if (now.ready) {
        if (now.signature !== last) {
          last = now.signature;
          since = Date.now();
        } else if (Date.now() - since >= config.stableMs) return now;
      } else last = null;
      if (Date.now() - started > config.settleTimeoutMs) throw new Error(`${label}: the screen did not settle after ${config.settleTimeoutMs} ms: ${now.why || "it kept changing"}`);
      await sleep(150);
    }
  }

  /**
   * Asks the runner for a picture of the screen once it is final. Looks again after the picture: if anything changed meanwhile (a late
   * thumbnail, a late frame) the picture is taken again, up to three times.
   */
  async function shoot(name, what) {
    for (let attempt = 1; attempt <= 3; attempt++) {
      const before = await settle(name);
      if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
      await sleep(200);
      const cell = q("[data-viewport]");
      await send("shot", {
        name,
        what,
        dpr: window.devicePixelRatio || 1,
        window: { screenX: window.screenX, screenY: window.screenY, innerWidth: window.innerWidth, innerHeight: window.innerHeight, outerWidth: window.outerWidth, outerHeight: window.outerHeight },
        view: { x: 0, y: 0, width: device(window.innerWidth), height: device(window.innerHeight) },
        photo: rectOf(cell),
        // Where photographs are drawn (the photo cell, the filmstrip, the look tiles), and the interface drawn over them (tags, chips):
        // the runner keeps the interface's exact colours when it shrinks the picture.
        photoRects: [...rectsOf("[data-viewport]"), ...rectsOf(".ym-filmstrip__thumb img"), ...rectsOf(".ym-look-tile img")],
        gradientRects: rectsOf(".ym-wheel__disc, .ym-histogram__box, .ym-curve__plot"),
        uiRects: rectsOf(".ym-photo-cell__chip, .ym-photo-cell__before, .ym-look-tile__badge, .ym-look-tile__default, .ym-filmstrip__picked"),
        photoId: currentId(),
        tab: q("[data-screen=photos]") ? activeTab() : "",
        status: textOf(".ym-status"),
      });
      const after = look();
      if (after.ready && after.signature === before.signature) return;
      note(`${name}: the screen changed while it was captured (${after.why || "new content"}); capturing again`);
    }
    throw new Error(`${name}: the screen kept changing under the capture`);
  }

  // --- Choosing photos ------------------------------------------------------------------------------------------------------------

  /** How colourful a thumbnail is (Hasler and Süsstrunk, 2003): the spread and the offset of the opponent channels. Null when it can't be read. */
  async function colourfulness(src) {
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = () => reject(new Error("the thumbnail did not load"));
      img.src = src;
    });
    const w = 96;
    const h = Math.max(1, Math.round((w * img.naturalHeight) / img.naturalWidth));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, w, h);
    const { data } = ctx.getImageData(0, 0, w, h);
    let n = 0, srg = 0, syb = 0, srg2 = 0, syb2 = 0;
    for (let i = 0; i < data.length; i += 4) {
      const rg = data[i] - data[i + 1];
      const yb = 0.5 * (data[i] + data[i + 1]) - data[i + 2];
      srg += rg; syb += yb; srg2 += rg * rg; syb2 += yb * yb; n++;
    }
    const mrg = srg / n, myb = syb / n;
    const sdrg = Math.sqrt(Math.max(0, srg2 / n - mrg * mrg));
    const sdyb = Math.sqrt(Math.max(0, syb2 / n - myb * myb));
    return Math.hypot(sdrg, sdyb) + 0.3 * Math.hypot(mrg, myb);
  }

  /** The photos that can be shown, in the order of the filmstrip: `{ id, name, thumb }`. */
  async function readyPhotos() {
    const state = await invoke("session_state");
    return state.photos
      .filter((p) => p.status && p.status.kind === "ready")
      .map((p) => ({ id: p.id, name: String(p.file_name).replace(/\.[^.]*$/, ""), file: p.file_name }));
  }

  /** The photo a `--pick` value names: `first`, `second`, `last`, `colourful`, a 1-based number, or part of a file name. */
  async function choose(spec, photos) {
    const s = String(spec || "first").trim().toLowerCase();
    if (s === "first") return photos[0];
    if (s === "second") return photos[1] || photos[0];
    if (s === "last") return photos[photos.length - 1];
    if (s === "colourful" || s === "colorful") {
      const scores = [];
      for (const photo of photos) {
        const thumb = q(`.ym-filmstrip__thumb[data-photo-id="${photo.id}"] img`);
        try {
          scores.push({ photo, score: thumb ? await colourfulness(thumb.src) : -1 });
        } catch (error) {
          note(`colourfulness of ${photo.name}: ${error.message}`);
          scores.push({ photo, score: -1 });
        }
      }
      note("colourfulness: " + scores.map((x) => `${x.photo.name} ${x.score.toFixed(1)}`).join(", "));
      scores.sort((a, b) => b.score - a.score);
      return scores[0].score >= 0 ? scores[0].photo : photos[0];
    }
    if (/^\d+$/.test(s)) return photos[Number(s) - 1];
    return photos.find((p) => p.file.toLowerCase() === s) || photos.find((p) => p.name.toLowerCase() === s) || photos.find((p) => p.file.toLowerCase().includes(s));
  }

  /** Opens a photo from the filmstrip and waits for its frame. */
  async function show(photo) {
    if (currentId() === photo.id) return;
    const before = frame() ? frame().src : "";
    const thumb = await waitFor("the filmstrip thumbnail", () => q(`button.ym-filmstrip__thumb[data-photo-id="${photo.id}"]`), 10000);
    thumb.click();
    await waitFor(`${photo.name} is the open photo`, () => currentId() === photo.id, 10000);
    await waitFor("the frame of the new photo", () => frame() && loaded(frame()) && frame().src !== before && frame().naturalWidth > 16, config.frameTimeoutMs);
  }

  /**
   * Draws the open photo once more (Before on, then off), so the render time in the header is that of a warm render: the first one of a
   * photo pays for its caches, and a README should show what the app does after that.
   */
  async function refresh() {
    for (let i = 0; i < 2; i++) {
      const before = frame() ? frame().src : "";
      key("\\");
      await waitFor("a new frame", () => frame() && loaded(frame()) && frame().src !== before, config.frameTimeoutMs);
    }
    await waitFor("Before is off again", () => !/Before · camera default/.test(textOf("[data-viewport]")), 10000);
  }

  async function openTab(name) {
    const tab = TABS[name];
    if (activeTab() === tab.title.toLowerCase()) return;
    key(tab.key, { metaKey: true });
    await waitFor(`the ${tab.title} tab`, () => activeTab() === tab.title.toLowerCase() && textOf(".ym-title-block__title").trim() === tab.title, 10000);
  }

  /**
   * Makes the webview 1440 x 900 CSS pixels, the size of the mockups. The window the app opens is 1440 x 900 with its menu bar inside
   * (25 px on Linux), so the page is 875 px tall; the page can't resize the window (the capabilities do not allow it), so the runner is
   * asked to (xdotool, on the X window). Without xdotool the page stays as it is and the runner says so.
   */
  async function fitWindow() {
    const want = { width: config.cssWidth, height: config.cssHeight };
    if (window.innerWidth === want.width && window.innerHeight === want.height) return;
    const dpr = window.devicePixelRatio || 1;
    const bars = Math.max(0, window.outerHeight - window.innerHeight);
    await send("fit", { width: Math.round(want.width * dpr), height: Math.round((want.height + bars) * dpr) });
    await waitFor("the window to take its new size", () => window.innerWidth === want.width && window.innerHeight === want.height, 8000).catch(() =>
      note(`the window stays ${window.innerWidth}x${window.innerHeight} CSS pixels instead of ${want.width}x${want.height} (is xdotool installed?)`),
    );
    await sleep(500);
  }

  // --- The run --------------------------------------------------------------------------------------------------------------------

  async function main() {
    await waitFor("Tauri", () => window.__TAURI_INTERNALS__, 15000);
    const info = await invoke("viewport_info");
    note(`viewport ${JSON.stringify(info)}, device scale ${window.devicePixelRatio}, window ${window.innerWidth}x${window.innerHeight} (screen ${window.screenX},${window.screenY}; outer ${window.outerWidth}x${window.outerHeight})`);
    if (!info || info.backend !== "fallback") throw new Error("the screenshots need the fallback viewport (the native one draws under the page); is this a Mac build? " + JSON.stringify(info));
    await waitFor("the Start screen", () => q("[data-screen=start]"), 30000);
    await fitWindow();

    // The Export panel names the folder it saves to. The default is next to the photos, which would put the runner's own paths on the
    // picture (a temporary folder, or the home folder of whoever ran this); the mockup's "~/Pictures/<folder> export" goes there instead.
    // Nothing is exported, and the preference lives in the runner's temporary app folder.
    if (config.exportShown) {
      const prefs = await invoke("preferences_get");
      await invoke("preferences_set", { patch: { export_defaults: Object.assign({}, prefs.export_defaults, { destination: config.exportShown }) } });
      await emit("looks_changed", null);
    }

    await dropPaths(config.paths);
    await waitFor("the Start screen reacts to the drop", () => /^Developing \d+ photos?$|photos? ready$|^No ARW files/.test(textOf("[data-screen=start] h2")) || q(".ym-error-bar"), 30000);
    if (q(".ym-error-bar")) throw new Error("the app said: " + textOf(".ym-error-bar").replace(/Dismiss$/, "").trim());
    if (/^No ARW files/.test(textOf("[data-screen=start] h2"))) throw new Error("the app found no ARW files in what was dropped");
    await waitFor("every photo developed", () => /^\d+ photos? ready$|^No photos ready$/.test(textOf("[data-screen=start] h2")), config.developTimeoutMs);
    if (/^No photos ready/.test(textOf("[data-screen=start] h2"))) throw new Error("no photo could be developed: " + textOf("[data-screen=start] main").replace(/\s+/g, " ").slice(0, 300));
    const skipped = qa("[data-screen=start] li").map((li) => li.textContent.replace(/\s+/g, " ")).filter((t) => /skipped/i.test(t));
    if (skipped.length) note("skipped: " + skipped.join(" | "));

    if (config.states.includes("start")) await shoot("start-ready", "the Start screen, every photo ready");

    const open = qa("button", q("[data-screen=start]")).find((b) => /Open photos/.test(b.textContent));
    if (!open) throw new Error("no Open photos button");
    open.click();
    await waitFor("the Photos screen", () => q("[data-screen=photos]"), 30000);
    await waitFor("the first frame", () => frame() && loaded(frame()) && frame().naturalWidth > 16, config.frameTimeoutMs);
    await waitFor("every filmstrip thumbnail", () => {
      const thumbs = qa(".ym-filmstrip__thumb").filter((t) => t.getAttribute("data-status") !== "failed");
      return thumbs.length > 0 && thumbs.every((t) => loaded(q("img", t)));
    }, config.settleTimeoutMs);

    const photos = await readyPhotos();
    note("photos: " + photos.map((p) => p.file).join(", "));
    const wanted = { look: "first", light: photos.length > 1 ? "second" : "first", colour: "colourful", detail: "first", export: "first" };
    // Each state in turn, the photo it asks for and the tab it shows; Export is last (it replaces the tool panel).
    for (const state of ["look", "light", "colour", "detail", "export"]) {
      if (!config.states.includes(state)) continue;
      const photo = await choose((config.pick && (config.pick[state] || config.pick.all)) || wanted[state], photos);
      if (!photo) throw new Error(`--pick for ${state}: no photo named ${(config.pick && config.pick[state]) || ""}`);
      await show(photo);
      await refresh();
      if (state === "export") {
        if (!q("aside[aria-label=Export]")) {
          key("e", { metaKey: true });
          await waitFor("the Export panel", () => q("aside[aria-label=Export]"), 10000);
        }
        await shoot("photo-export", `the Export panel over ${photo.name}`);
      } else {
        await openTab(state);
        await shoot(state === "look" ? "photo-look" : state === "light" ? "photo-light" : state === "colour" ? "photo-colour" : "photo-detail", `${photo.name}, ${TABS[state].title} tab`);
      }
    }
  }

  main().then(
    () => send("done", { ok: true }),
    (error) => send("done", { ok: false, error: error && error.message ? error.message : String(error) }),
  );
})();
