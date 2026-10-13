// The helpers that every driver of the real shell shares. Runs INSIDE the real webview (see host.mjs, which serves it before the driver),
// plain browser JavaScript, no imports. A driver reads what it needs from `window.__YM_KIT__`:
//
//   const { invoke, sleep, send, waitFor, q, qa, ... } = window.__YM_KIT__;
//
// Used by `e2e/real/driver.js` (the smoke test of `npm run e2e:shell`) and `tools/readme/driver.js` (the README screenshots).
(function () {
  "use strict";
  const invoke = (command, args) => window.__TAURI_INTERNALS__.invoke(command, args);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /**
   * Sends a report to the runner as an image request (`/__report?...`): the page's connect-src does not matter for those. A long payload
   * is cut into pieces a URL can carry; each piece is encoded on its own (cut by character, never inside one). Resolves when the runner
   * has answered the last piece (it answers after it has acted on the report, so a screenshot is taken before the driver goes on), or
   * after five seconds.
   */
  function send(kind, payload) {
    const characters = Array.from(JSON.stringify(payload));
    const id = Math.random().toString(36).slice(2);
    const size = 400;
    const pieces = Math.max(1, Math.ceil(characters.length / size));
    const answers = [];
    for (let i = 0; i < pieces; i++) {
      const piece = encodeURIComponent(characters.slice(i * size, (i + 1) * size).join(""));
      answers.push(
        new Promise((resolve) => {
          const image = new Image();
          // The runner answers 204: an image request that "fails" is the answer.
          image.onload = image.onerror = () => resolve();
          image.src = `/__report?kind=${kind}&id=${id}&i=${i}&n=${pieces}&data=${piece}`;
        }),
      );
    }
    return Promise.race([Promise.all(answers), sleep(5000)]);
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
  /** The fallback viewport's picture of the photo cell. */
  const frame = () => q("[data-viewport] img.ym-photo-cell__frame");
  /** A card (a section named by its title), by its title. */
  const region = (name) =>
    qa("section[aria-labelledby]").find((s) => {
      const title = document.getElementById(s.getAttribute("aria-labelledby"));
      return title && title.textContent.trim().toLowerCase() === name.toLowerCase();
    });
  const zoomLabel = () => (qa("header button").find((b) => /^(Fit|\d+%)/.test(b.textContent.trim())) || { textContent: "" }).textContent.trim();

  /** Sends a Tauri event from the page, as the shell would (a drop of files, the news that the looks changed). */
  const emit = (event, payload) => invoke("plugin:event|emit", { event, payload });

  /**
   * Drops files or folders on the window, as the shell sends it: Tauri's own `tauri://drag-drop` event, with real paths. The Start screen
   * starts listening for drops a moment after it appears (the listener is registered through the bridge), and an event sent before that
   * is lost, as it would be for a person who drops that fast; the drop is sent again every few seconds until the Start screen reacts.
   * Opening the same files twice is harmless (the core knows them).
   */
  async function dropPaths(paths) {
    // (An error in the bar at the foot of the window is a reaction too: sending the same drop again would not change it.)
    const reacts = () => /^Developing \d+ photos?$|photos? ready$|^No ARW files/.test(textOf("[data-screen=start] h2")) || !!q(".ym-error-bar");
    for (let attempt = 0; attempt < 10 && !reacts(); attempt++) {
      await emit("tauri://drag-drop", { paths, position: { x: 10, y: 10 } });
      const until = Date.now() + 4000;
      while (Date.now() < until && !reacts()) await sleep(100);
    }
  }

  window.__YM_KIT__ = { invoke, sleep, send, waitFor, q, qa, loaded, textOf, key, frame, region, zoomLabel, emit, dropPaths };
})();
