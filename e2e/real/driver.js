// Runs INSIDE the real webview of the real Tauri shell (see shell-smoke.mjs, which serves it with the built frontend).
// Plain browser JavaScript, no imports. It drives the real React app through the DOM and the shell's own commands, checks what it
// sees, and sends every result to the runner as an image request (`/__report?...`): the page's connect-src does not matter for those.
//
// The folder is opened the way a person does it: by dropping it on the window (Tauri's own `tauri://drag-drop` event, with real paths), so
// the store hears every event from the start. The checks then run on the Start screen, the Photos screen and the Export panel.
(function () {
  "use strict";
  const config = window.__SMOKE__;
  const invoke = (command, args) => window.__TAURI_INTERNALS__.invoke(command, args);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const results = [];

  function send(kind, payload) {
    // A long payload is cut into pieces a URL can carry; each piece is encoded on its own (cut by character, never inside one).
    const characters = Array.from(JSON.stringify(payload));
    const id = Math.random().toString(36).slice(2);
    const size = 400;
    const pieces = Math.max(1, Math.ceil(characters.length / size));
    for (let i = 0; i < pieces; i++) {
      const piece = encodeURIComponent(characters.slice(i * size, (i + 1) * size).join(""));
      new Image().src = `/__report?kind=${kind}&id=${id}&i=${i}&n=${pieces}&data=${piece}`;
    }
  }

  function record(name, ok, detail) {
    const entry = { name, ok: !!ok, detail: detail === undefined ? "" : String(detail) };
    results.push(entry);
    send("check", entry);
  }

  async function waitFor(label, test, timeoutMs) {
    const started = Date.now();
    for (;;) {
      let value;
      try {
        value = await test();
      } catch (error) {
        value = undefined;
      }
      if (value) return value;
      if (Date.now() - started > timeoutMs) throw new Error(`timed out after ${timeoutMs} ms: ${label}`);
      await sleep(100);
    }
  }

  const q = (selector, root) => (root || document).querySelector(selector);
  const qa = (selector, root) => [...(root || document).querySelectorAll(selector)];
  const loaded = (img) => img && img.complete && img.naturalWidth > 0;
  const textOf = (selector) => (q(selector) ? q(selector).textContent : "");
  const key = (k, extra) => window.dispatchEvent(new KeyboardEvent("keydown", Object.assign({ key: k, bubbles: true, cancelable: true }, extra || {})));
  const frame = () => q("[data-viewport] img.ym-photo-cell__frame");
  /** A card (a section named by its title), by its title. */
  const region = (name) =>
    qa("section[aria-labelledby]").find((s) => {
      const title = document.getElementById(s.getAttribute("aria-labelledby"));
      return title && title.textContent.trim().toLowerCase() === name.toLowerCase();
    });
  const zoomLabel = () => (qa("header button").find((b) => /^(Fit|\d+%)/.test(b.textContent.trim())) || { textContent: "" }).textContent.trim();

  window.addEventListener("error", (e) => record("no uncaught error", false, e.message));
  /** While a check expects a promise to be rejected (the drag script of Tauri does not wait for its commands), rejections are collected here. */
  let rejections = null;
  window.addEventListener("unhandledrejection", (e) => {
    const reason = String(e.reason && e.reason.message ? e.reason.message : typeof e.reason === "string" ? e.reason : JSON.stringify(e.reason));
    if (rejections) rejections.push(reason);
    else record("no unhandled rejection", false, reason);
  });

  /** Sends a Tauri event from the page, as the shell would (a drop of files, the news that the looks changed). */
  const emit = (event, payload) => invoke("plugin:event|emit", { event, payload });

  /**
   * Drops a folder on the window, as the shell sends it. The Start screen starts listening for drops a moment after it appears (the
   * listener is registered through the bridge), and an event sent before that is lost, as it would be for a person who drops that fast;
   * the drop is sent again every few seconds until the Start screen reacts. Opening the same files twice is harmless (the core knows them).
   */
  async function dropFolder(path) {
    const reacts = () => /^Developing \d+ photos?$|photos? ready$|^No ARW files/.test(textOf("[data-screen=start] h2"));
    for (let attempt = 0; attempt < 10 && !reacts(); attempt++) {
      await emit("tauri://drag-drop", { paths: [path], position: { x: 10, y: 10 } });
      const until = Date.now() + 4000;
      while (Date.now() < until && !reacts()) await sleep(100);
    }
  }

  async function photos() {
    const info = await invoke("viewport_info");
    record("viewport_info answers", info && (info.backend === "native" || info.backend === "fallback"), JSON.stringify(info));
    await waitFor("the Start screen", () => q("[data-screen=start]"), 30000);

    // The remembered export folder is the runner's, so nothing lands next to the photos. The store reads the preferences again when the looks change.
    const prefs = await invoke("preferences_get");
    await invoke("preferences_set", { patch: { export_defaults: Object.assign({}, prefs.export_defaults, { destination: config.exportDir }) } });
    await emit("looks_changed", null);
    if (config.only === "geometry") {
      // `YM_SMOKE_ONLY=geometry`: just the failure-path folder with its geometry checks, for a quick run.
      if (!config.brokenDir) throw new Error("the geometry checks need the synthetic files of testdata/synthetic");
      await brokenFile(info.backend);
      finish();
      return;
    }

    // 0. Drop the folder on the window: Tauri's own drag-and-drop event, with real paths, as the shell sends it.
    const t0 = Date.now();
    await dropFolder(config.dir);
    await waitFor("the Start screen develops the photos", () => /^Developing \d+ photos?$/.test(textOf("[data-screen=start] h2")) || /photos? ready$/.test(textOf("[data-screen=start] h2")), 30000);
    record("a drop of the folder opens the session and the Start screen shows the progress of photo_progress", true, `${textOf("[data-screen=start] h2")} after ${Date.now() - t0} ms; tags ${[...new Set(qa("[data-screen=start] .ym-tag").map((t) => t.textContent.trim()))].join(", ")}`);
    await waitFor("every photo developed", () => /^\d+ photos? ready$/.test(textOf("[data-screen=start] h2")), config.developTimeoutMs);
    const tags = qa("[data-screen=start] .ym-tag").map((t) => t.textContent.trim());
    record("the Start screen ends with every photo Ready or Skipped", tags.length > 0 && tags.every((t) => /^(ready|skipped)$/i.test(t)), tags.join(", ") + "; " + textOf("[data-screen=start] h2"));
    const open = qa("button", q("[data-screen=start]")).find((b) => /Open photos/.test(b.textContent));
    record("Open photos is there", !!open, "");
    open.click();
    await waitFor("the Photos screen", () => q("[data-screen=photos]"), 30000);
    const cell = await waitFor("the photo cell", () => q("[data-viewport]"), 10000);
    const backend = await waitFor("viewport_info reached the store", () => cell.getAttribute("data-viewport-backend"), 10000);
    record("the cell knows the backend", backend === "native" || backend === "fallback", backend);
    const bg = getComputedStyle(cell).backgroundColor;
    record("the cell is " + (backend === "native" ? "transparent" : "paper"), backend === "native" ? bg === "rgba(0, 0, 0, 0)" : bg !== "rgba(0, 0, 0, 0)", bg);

    // 1. The picture.
    if (backend === "fallback") {
      const box = cell.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const img = await waitFor("a frame of the cell's size", () => frame() && loaded(frame()) && frame().naturalWidth > 16 && frame(), config.frameTimeoutMs);
      record("a frame shows, from the ym-frame scheme", img.src.startsWith("ym-frame://") || img.src.startsWith("http://ym-frame.localhost/"), img.src);
      record("the frame is the cell's size in device pixels", Math.abs(img.naturalWidth - Math.round(box.width * dpr)) <= 1 && Math.abs(img.naturalHeight - Math.round(box.height * dpr)) <= 1, `${img.naturalWidth}x${img.naturalHeight} for a cell of ${box.width.toFixed(1)}x${box.height.toFixed(1)} at ${dpr}x`);
      const shown = img.getBoundingClientRect();
      record("the frame lies on the cell, one picture pixel to one device pixel", Math.abs(shown.width - box.width) <= 1 && Math.abs(shown.left - box.left) <= 1 && Math.abs(shown.top - box.top) <= 1, `${shown.width.toFixed(1)}x${shown.height.toFixed(1)} at ${shown.left.toFixed(1)},${shown.top.toFixed(1)}`);
    }
    await waitFor("render_stats in the header", () => /· \d+ ms ·/.test(textOf(".ym-status")), 30000);
    record("the header says how the photo was drawn", true, textOf(".ym-status"));

    // 2. Thumbnails of the filmstrip (thumbnail_ready) and the Look tab's tiles (look_thumbnail).
    const strip = qa(".ym-filmstrip__thumb img");
    await waitFor("filmstrip thumbnails", () => qa(".ym-filmstrip__thumb img").filter(loaded).length >= 1, 30000);
    record("the filmstrip shows the thumbnails of thumbnail_ready", qa(".ym-filmstrip__thumb img").filter(loaded).length >= 1, `${qa(".ym-filmstrip__thumb img").filter(loaded).length} of ${qa(".ym-filmstrip__thumb").length}`);
    // The event carries an address, not the picture: the shell serves it on the `ym-frame` scheme (`docs/01` IPC).
    const thumbSources = qa(".ym-filmstrip__thumb img").filter(loaded).map((img) => img.src);
    record(
      "the thumbnails load from the ym-frame scheme, as addresses and not as bytes of the event",
      thumbSources.length >= 1 && thumbSources.every((src) => /^(ym-frame:\/\/localhost|http:\/\/ym-frame\.localhost)\/thumb-\d+-\d+\.png$/.test(src)),
      thumbSources[0] || "no thumbnail",
    );
    await waitFor("look tiles", () => qa(".ym-look-tile img").filter(loaded).length >= 1, config.lookTimeoutMs);
    record("the Look tab's tiles come from look_thumbnail", true, `${qa(".ym-look-tile img").filter(loaded).length} tiles drawn`);
    const automatic = region("Automatic");
    record("the Automatic card has the photo's values", !!automatic && /EV/.test(automatic.textContent), automatic ? automatic.textContent.replace(/\s+/g, " ").slice(0, 160) : "no card");
    void strip;

    // 3. Pointer input over the photo.
    const rect = cell.getBoundingClientRect();
    const at = (u, v) => ({ clientX: rect.left + rect.width * u, clientY: rect.top + rect.height * v, bubbles: true });
    const before = frame() ? frame().src : "";
    // A photo with fewer pixels than the cell has (the synthetic ones) is already shown at more than 100 % at Fit, and the core draws Fit
    // for a double click on it: the header must say so, not "100%".
    const fitNow = Number((/^Fit (\d+)%/.exec(zoomLabel()) || [])[1]);
    cell.dispatchEvent(new MouseEvent("dblclick", Object.assign(at(0.5, 0.5), { detail: 2 })));
    if (fitNow >= 100) {
      await sleep(1500);
      record("a double click on a photo smaller than the cell stays at Fit, as the core draws it", zoomLabel().startsWith("Fit"), `${zoomLabel()} (Fit was ${fitNow}%)`);
    } else {
      await waitFor("the header says 100%", () => zoomLabel().startsWith("100%"), 10000);
      record("a double click goes to 100 % in the header", true, zoomLabel());
      if (backend === "fallback") {
        await waitFor("a new frame at 100 %", () => frame() && loaded(frame()) && frame().src !== before, config.frameTimeoutMs);
        record("a double click makes the shell send a new frame", true, frame().src);
      }
      cell.dispatchEvent(new MouseEvent("dblclick", Object.assign(at(0.5, 0.5), { detail: 2 })));
      await waitFor("the header says Fit again", () => zoomLabel().startsWith("Fit"), 10000);
      record("a second double click is back to Fit", true, zoomLabel());
    }
    if (backend === "fallback") await sleep(1500);

    cell.dispatchEvent(new PointerEvent("pointermove", Object.assign(at(0.5, 0.45), { altKey: true, pointerId: 1 })));
    const readout = await waitFor("the pixel readout", () => /^R \d+ G \d+ B \d+$/.test(textOf(".ym-photo-cell__readout")) && textOf(".ym-photo-cell__readout"), 15000);
    record("holding ⌥ reads the pixel from sample_pixel", true, readout);
    cell.dispatchEvent(new PointerEvent("pointerleave", { bubbles: false, pointerId: 1 }));

    // 4. A setting changed through the shell: a new frame and new numbers in the header.
    const photo = (await invoke("session_state")).photos[(await invoke("session_state")).current_index];
    const frameBefore = frame() ? frame().src : "";
    await invoke("update_settings", { photoId: photo.id, patch: { exposure: { kind: "manual", ev: 1.0 } } });
    if (backend === "fallback") {
      await waitFor("a frame after the settings changed", () => frame() && loaded(frame()) && frame().src !== frameBefore, config.frameTimeoutMs);
      record("a settings change makes the shell send a new frame", true, frame().src);
    }
    await invoke("update_settings", { photoId: photo.id, patch: { exposure: { kind: "auto" } } });
    if (backend === "fallback") await sleep(1500);

    // 4a. Before (`\`): the camera view, before the look and the display transform; the label says so, and the shell draws it.
    {
      const shown = frame() ? frame().src : "";
      key("\\");
      await waitFor("the Before label", () => /Before · camera log/.test(textOf("[data-viewport]")), 10000);
      if (backend === "fallback") await waitFor("a frame of the camera view", () => frame() && loaded(frame()) && frame().src !== shown, config.frameTimeoutMs);
      const camera = await invoke("sample_pixel", { x: 300, y: 200 }).then((v) => v, () => null);
      key("\\");
      await waitFor("the label is gone", () => !/Before · camera log/.test(textOf("[data-viewport]")), 10000);
      if (backend === "fallback") await waitFor("the picture is back", () => frame() && loaded(frame()) && frame().src !== shown, config.frameTimeoutMs);
      const developed = await invoke("sample_pixel", { x: 300, y: 200 }).then((v) => v, () => null);
      record("Before shows the camera view: a new frame, and not the developed pixels", !!camera && !!developed && (camera.r !== developed.r || camera.g !== developed.g || camera.b !== developed.b), `camera ${camera ? camera.r + "," + camera.g + "," + camera.b : "none"}, developed ${developed ? developed.r + "," + developed.g + "," + developed.b : "none"}`);
    }
    if (backend === "fallback") await sleep(1500);

    // 4b. Where a point of the cell lands on the photo, off the middle (units, letterbox, zoom about the pointer, pan).
    await geometry(backend, true);

    // 5. Pick white through the shell.
    key("w");
    const layer = await waitFor("the pick-white layer", () => q(".ym-pick-white"), 5000);
    const lb = layer.getBoundingClientRect();
    layer.dispatchEvent(new MouseEvent("click", { clientX: lb.left + lb.width * config.pickAt[0], clientY: lb.top + lb.height * config.pickAt[1], bubbles: true }));
    const outcome = await waitFor("a pick or an error", () => {
      const edited = qa(".ym-tag", region("White balance")).some((t) => /edited/i.test(t.textContent));
      const bar = q(".ym-error-bar");
      return edited ? { picked: true } : bar ? { picked: false, error: bar.textContent } : null;
    }, 60000);
    record("Pick white: the core picked the spot, or said why not in words", outcome.picked || /\.$/.test((outcome.error || "").replace(/Dismiss$/, "").trim()), outcome.picked ? "picked: the White balance card says EDITED" : "refused: " + outcome.error);
    if (outcome.picked) {
      const reset = qa("button", region("White balance")).find((b) => /^reset$/i.test(b.textContent.trim()));
      if (reset) reset.click();
      await waitFor("the pick is cleared", () => !qa(".ym-tag", region("White balance")).some((t) => /edited/i.test(t.textContent)), 30000);
      record("Reset clears the pick", true, "");
    } else {
      const dismiss = qa(".ym-error-bar button").find((b) => /dismiss/i.test(b.textContent));
      if (dismiss) dismiss.click();
    }

    // 6. An export, with the update cell held back.
    key("e", { metaKey: true });
    const panel = await waitFor("the Export panel", () => q("aside[aria-label=Export]"), 10000);
    const go = await waitFor("the Export button", () => q(".ym-tools__footer button", panel), 10000);
    await waitFor("the estimate", () => /≈/.test(go.textContent), 15000).catch(() => undefined);
    // Never export anywhere but the runner's folder: the panel has to say it is going there.
    const folderName = config.exportDir.split("/").pop();
    const goesThere = panel.textContent.includes(folderName);
    record("the Export panel saves to the remembered export folder", goesThere, goesThere ? folderName : "the panel does not name " + folderName + ": " + panel.textContent.replace(/\s+/g, " ").slice(0, 300));
    if (!goesThere) throw new Error("the export would not go to the runner's folder");
    go.click();
    const busyDuring = await waitFor("export_busy", async () => (await invoke("export_busy")) === true, 15000).then(() => true, () => false);
    record("the core says an export is busy while it runs", busyDuring, "");
    const finished = await waitFor("the export finishes", () => {
      const h = q("aside[aria-label=Export] h2");
      return h && /^(Exported|Not exported)$/.test(h.textContent.trim()) && h.textContent.trim();
    }, config.exportTimeoutMs);
    record("the export panel follows export_progress to the end", finished === "Exported", `${finished}: ${textOf("aside[aria-label=Export] .ym-title-block")}`.replace(/\s+/g, " ").slice(0, 220));
    record("the core is quiet again", (await invoke("export_busy")) === false, "");

    // 7. The header drags the window (titleBarStyle Overlay has no title bar): Tauri's drag script asks the shell on a press of the folder cell
    // (and, on a double click, to maximise); each command has to be allowed by the capabilities of the window.
    const brand = q("header .ym-brand");
    rejections = [];
    const press = (detail) => brand.querySelector(".ym-brand-name").dispatchEvent(new MouseEvent("mousedown", { button: 0, detail, bubbles: true, cancelable: true, composed: true }));
    press(1);
    await sleep(800);
    record("a press on the folder cell may start dragging the window", rejections.length === 0, rejections.join(" | "));
    rejections = [];
    press(2);
    await sleep(800);
    record("a double click on the folder cell may maximise the window", rejections.length === 0, rejections.join(" | "));
    rejections = null;

    // 8. Close the session: the Start screen, and the shell takes the picture away.
    send("screenshot", { name: "photos" });
    await sleep(1500);
    key("w", { metaKey: true });
    await waitFor("the Start screen", () => q("[data-screen=start]"), 15000);
    record("closing the session returns to Start", true, "");
    const closed = await invoke("session_state");
    record("the core's session is empty", closed.photos.length === 0, "");
    if (config.brokenDir) await brokenFile(backend);
    finish();
  }

  /** A folder with two good files, a file that is not a raw file at all, and a text file: the failure paths of the shell. */
  async function brokenFile(backend) {
    await dropFolder(config.brokenDir);
    await waitFor("the Start screen settles", () => /photos? ready$/.test(textOf("[data-screen=start] h2")), config.developTimeoutMs);
    const tags = qa("[data-screen=start] .ym-tag").map((t) => t.textContent.trim().toLowerCase());
    record("a file that can't be read is Skipped on the Start screen, the rest Ready", tags.includes("skipped") && tags.includes("ready"), `${tags.join(", ")}; ${textOf("[data-screen=start] main").replace(/\s+/g, " ").slice(0, 300)}`);
    const reasons = qa("[data-screen=start] li").map((li) => li.textContent.replace(/\s+/g, " ")).filter((t) => /skipped/i.test(t));
    record("the Skipped file says why, in words", reasons.length > 0 && /\.\s*$|[a-z]/.test(reasons[0]), reasons[0] || "no row");
    qa("button", q("[data-screen=start]")).find((b) => /Open photos/.test(b.textContent)).click();
    await waitFor("the Photos screen", () => q("[data-screen=photos]"), 30000);
    if (backend === "fallback") await waitFor("a frame of the first photo", () => frame() && loaded(frame()) && frame().naturalWidth > 16, config.frameTimeoutMs);
    await geometry(backend, false);
    const bad = await waitFor("the can't-open thumbnail", () => qa(".ym-filmstrip__thumb").find((b) => /can't open/i.test(b.getAttribute("aria-label") || "")), 10000);
    bad.click();
    await waitFor("the note over the photo", () => /can't be opened/.test(textOf(".ym-photo-note")), 10000);
    record("selecting the broken file says so over the photo", true, textOf(".ym-photo-note").replace(/\s+/g, " ").slice(0, 200));
    if (backend === "fallback") {
      await waitFor("the previous photo's picture is gone", () => frame() === null, 15000);
      record("the previous photo's picture does not stay behind it", true, "");
    }
    const good = qa(".ym-filmstrip__thumb").find((b) => !/can't open/i.test(b.getAttribute("aria-label") || ""));
    good.click();
    await waitFor("the note is gone", () => !q(".ym-photo-note"), 10000);
    if (backend === "fallback") {
      await waitFor("the picture is back for a good photo", () => frame() && loaded(frame()) && frame().naturalWidth > 16, config.frameTimeoutMs);
      record("a good photo shows its picture again", true, frame().src);
    }
    key("w", { metaKey: true });
    await waitFor("the Start screen", () => q("[data-screen=start]"), 15000);
  }

  /**
   * Where a point of the cell lands on the photo, in the real core: the units (CSS pixels of the cell, at the scale of the display), the
   * letterbox, the zoom about the pointer and the pan. A point off the middle of a photo is the only thing that can tell a wrong scale or a
   * missing letterbox from a right answer: the middle of the cell is the middle of the photo whatever is wrong. `zooming` is false for
   * the tiny synthetic photos (a 512 px photo is never shown at 100 % in a cell of 1014 px), which only take the Pick white check.
   */
  async function geometry(backend, zooming) {
    const cell = q("[data-viewport]");
    const dpr = window.devicePixelRatio || 1;
    const state = await invoke("session_state");
    const photo = state.photos[state.current_index];
    const sample = (x, y) => invoke("sample_pixel", { x, y }).then((v) => v, () => null);
    const distance = (a, b) => (a && b ? Math.max(Math.abs(a.r - b.r), Math.abs(a.g - b.g), Math.abs(a.b - b.b)) : Infinity);
    const show = (v) => (v ? `${v.r},${v.g},${v.b}` : "none");
    const rect = () => cell.getBoundingClientRect();
    const client = (p) => ({ clientX: rect().left + p.x, clientY: rect().top + p.y });

    // 1. Pick white off the middle of the picture, at Fit (the picture is letterboxed when its shape is not the cell's): the pick has to land
    // where the click was, as a share of the photo as the camera oriented it, however the photo is turned and cropped. The core may refuse
    // a spot that is too bright or too dark; the next spot is tried then.
    const box = rect();
    const where = `${photo.width_px}x${photo.height_px} photo in a ${box.width.toFixed(0)}x${box.height.toFixed(0)} cell at ${dpr}x`;
    /** Clicks the picture of aspect `aspect` at the shares `(u, v)` of it, in Pick white mode; resolves with the `white_pick` the core set. */
    async function pickAt(aspect, spots) {
      const cellAspect = box.width / box.height;
      const picture = aspect >= cellAspect ? { w: box.width, h: box.width / aspect } : { w: box.height * aspect, h: box.height };
      picture.x = (box.width - picture.w) / 2;
      picture.y = (box.height - picture.h) / 2;
      for (const [u, v] of spots) {
        key("w");
        const layer = await waitFor("the pick-white layer", () => q(".ym-pick-white"), 5000);
        const lb = layer.getBoundingClientRect();
        layer.dispatchEvent(new MouseEvent("click", { clientX: lb.left + picture.x + picture.w * u, clientY: lb.top + picture.y + picture.h * v, bubbles: true }));
        const outcome = await waitFor("a pick or an error", () => {
          const edited = qa(".ym-tag", region("White balance")).some((t) => /edited/i.test(t.textContent));
          const bar = q(".ym-error-bar");
          return edited ? { picked: true } : bar ? { picked: false } : null;
        }, 60000);
        if (outcome.picked) {
          const got = (await invoke("get_settings", { photoId: photo.id })).settings.white_pick;
          const reset = qa("button", region("White balance")).find((b) => /^reset$/i.test(b.textContent.trim()));
          if (reset) reset.click();
          await waitFor("the pick is cleared", () => !qa(".ym-tag", region("White balance")).some((t) => /edited/i.test(t.textContent)), 30000);
          return { u, v, got };
        }
        const dismiss = qa(".ym-error-bar button").find((b) => /dismiss/i.test(b.textContent));
        if (dismiss) dismiss.click();
      }
      return null;
    }
    /** The pick the core should have set for a click at `(u, v)` of the picture, as a function of them. */
    function judge(name, measured, expected) {
      if (!measured || !measured.got) {
        record(name, !measured, measured ? "no white_pick" : `the core refused every spot tried (nothing to compare); ${where}`);
        return;
      }
      const want = expected(measured.u, measured.v);
      const off = Math.max(Math.abs(measured.got[0] - want[0]), Math.abs(measured.got[1] - want[1]));
      record(name, off < 0.01, `clicked ${measured.u.toFixed(3)},${measured.v.toFixed(3)}: wanted ${want.map((n) => n.toFixed(3))}, got ${measured.got.map((n) => n.toFixed(3))} (${where})`);
    }
    /** Changes the photo's settings through the shell, tells the store to read them again, and waits for the picture to follow. */
    async function change(patch) {
      const frameBefore = frame() ? frame().src : "";
      await invoke("update_settings", { photoId: photo.id, patch });
      await emit("settings_changed", { photo_id: photo.id });
      if (backend === "fallback") await waitFor("a new frame", () => frame() && loaded(frame()) && frame().src !== frameBefore, config.frameTimeoutMs);
      await sleep(800);
    }
    const W = photo.width_px;
    const H = photo.height_px;
    const spots = [[0.3, 0.2], [0.7, 0.65], [0.15, 0.8], [0.85, 0.3]];
    judge("Pick white lands where the click was, off the middle of the photo", await pickAt(W / H, spots), (u, v) => [u, v]);
    // The photo turned a quarter turn clockwise: the point of the picture (u, v) is the point (v, 1 - u) of the photo as the camera had it.
    await change({ orientation: 90 });
    judge("Pick white on a photo turned 90°: the core maps the click back through the turn", await pickAt(H / W, spots), (u, v) => [v, 1 - u]);
    // Cropped (the frame is a rectangle of the picture as it is turned): (u, v) of the frame is (x + u w, y + v h) of that picture.
    await change({ orientation: 0, crop: { rect: { x: 0.25, y: 0.2, width: 0.5, height: 0.6 }, angle_deg: 0, aspect: "free" } });
    judge("Pick white on a cropped photo: the core maps the click back through the crop", await pickAt((0.5 * W) / (0.6 * H), spots), (u, v) => [0.25 + 0.5 * u, 0.2 + 0.6 * v]);
    // Both: the crop is a rectangle of the turned picture, and the turn is undone after the crop.
    await change({ orientation: 90, crop: { rect: { x: 0.1, y: 0.2, width: 0.5, height: 0.4 }, angle_deg: 0, aspect: "free" } });
    judge(
      "Pick white on a turned and cropped photo: the crop, then the turn",
      await pickAt((0.5 * H) / (0.4 * W), spots),
      (u, v) => {
        const cx = 0.1 + 0.5 * u;
        const cy = 0.2 + 0.4 * v;
        return [cy, 1 - cx];
      },
    );
    await change({ orientation: 0, crop: null });
    if (!zooming) return;

    // 2. Zoom and pan need a photo that is shown at 100 % in this cell: its pixels have to outnumber the cell's.
    const fit = Number((/^Fit (\d+)%/.exec(zoomLabel()) || [])[1]);
    if (!(fit > 0 && fit < 100)) {
      record("zooming in about the pointer and panning", true, `skipped: the photo is too small for 100 % in this cell (${zoomLabel() || "no label"}; ${where})`);
      return;
    }
    if (backend === "fallback") await sleep(1500);

    // 2a. A wheel zoom with ⌘ about a point keeps the picture under that point: the pixel there at 100 % and at twice that agree, give or
    // take what the finer sampling of a textured picture changes. 100 % first, from a double click at the point.
    const P = { x: Math.round(box.width * 0.4), y: Math.round(box.height * 0.45) };
    cell.dispatchEvent(new MouseEvent("dblclick", Object.assign(client(P), { detail: 2, bubbles: true })));
    await waitFor("the header says 100%", () => zoomLabel().startsWith("100%"), 10000);
    const before = await sample(P.x, P.y);
    cell.dispatchEvent(new WheelEvent("wheel", Object.assign(client(P), { deltaY: -69.3, ctrlKey: true, bubbles: true, cancelable: true })));
    await waitFor("the header says about 200%", () => /^(19\d|20\d)%/.test(zoomLabel()), 10000);
    const after = await sample(P.x, P.y);
    record("a ⌘ wheel zoom keeps the picture under the pointer", distance(before, after) <= 24, `${show(before)} at 100 %, ${show(after)} at ${zoomLabel()} (${dpr}x)`);
    await backTo100(cell, client, box);

    // 2b. Pan: at 100 % every pixel of the picture is its own, so a drag moves the picture with the pointer exactly when the pixel that was
    // under the start of the drag is under its end afterwards (and the one that was under the end is not).
    const from = { x: Math.round(box.width * 0.4), y: Math.round(box.height * 0.4) };
    const to = { x: from.x + 90, y: from.y + 50 };
    const here = await sample(from.x, from.y);
    const there = await sample(to.x, to.y);
    cell.dispatchEvent(new PointerEvent("pointerdown", Object.assign(client(from), { button: 0, bubbles: true, pointerId: 1 })));
    for (let i = 1; i <= 6; i++) window.dispatchEvent(new PointerEvent("pointermove", Object.assign(client({ x: from.x + ((to.x - from.x) * i) / 6, y: from.y + ((to.y - from.y) * i) / 6 }), { bubbles: true, pointerId: 1 })));
    window.dispatchEvent(new PointerEvent("pointerup", Object.assign(client(to), { button: 0, bubbles: true, pointerId: 1 })));
    const moved = await sample(to.x, to.y);
    const informative = distance(here, there) >= 4;
    record(
      "a drag pans the picture with the pointer, in CSS pixels at any display scale",
      !informative || (distance(moved, here) <= 2 && distance(moved, there) >= 2),
      `${informative ? "" : "(the two spots hardly differ: nothing to tell) "}under the start ${show(here)}, under the end before ${show(there)}, after the drag ${show(moved)} (${dpr}x)`,
    );
    key("z");
    await waitFor("the header says Fit again", () => zoomLabel().startsWith("Fit"), 10000);
    if (backend === "fallback") await sleep(1500);
  }

  /** Back to exactly 100 % (the zoom menu's choice, which the core takes as it is) after a wheel zoom. */
  async function backTo100(cell, client, box) {
    key("z");
    await waitFor("Fit", () => zoomLabel().startsWith("Fit"), 10000);
    cell.dispatchEvent(new MouseEvent("dblclick", Object.assign(client({ x: box.width * 0.5, y: box.height * 0.5 }), { detail: 2, bubbles: true })));
    await waitFor("100%", () => zoomLabel().startsWith("100%"), 10000);
  }

  function finish() {
    send("done", { failed: results.filter((r) => !r.ok).length, total: results.length });
  }

  async function main() {
    try {
      await waitFor("Tauri", () => window.__TAURI_INTERNALS__, 15000);
      await photos();
    } catch (error) {
      record("the smoke test ran to the end", false, error && error.message ? error.message : String(error));
      send("screenshot", { name: "failure" });
      await sleep(1500);
      finish();
    }
  }
  main();
})();
