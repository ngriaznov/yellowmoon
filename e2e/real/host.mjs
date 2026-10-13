// The Node side that every driver of the real shell shares: the server that hands the built frontend to the debug Tauri app (which loads its
// `devUrl`, http://localhost:1420) with a driver added to index.html, an Xvfb screen to run it on, the app itself, and a screenshot of the
// screen. Used by `e2e/real/shell-smoke.mjs` (`npm run e2e:shell`) and `tools/readme/screenshots.mjs` (`npm run readme:screenshots`).
//
// The driver runs inside the webview and reports back as image requests (`/__report?kind=...`); `serveFrontend` collects the pieces and
// calls `onReport(kind, payload)`, and answers the last piece only after `onReport` is done (a driver can wait for a screenshot).
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

/** The port the debug app loads its frontend from (`build.devUrl` of `tauri.conf.json`). */
export const PORT = 1420;
/** The repository's root. */
export const ROOT = resolve(import.meta.dirname, "../..");

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".woff2": "font/woff2", ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json", ".ico": "image/x-icon" };

/**
 * Serves `dist` with `driver` (the source of a plain browser script) added to index.html after `kit.js` and `window.__SMOKE__ = config`.
 * `onReport(kind, payload)` hears the driver's reports; it may return a promise, and the driver's request is answered when it settles.
 * Does not listen yet: see `listen`.
 */
export function serveFrontend({ dist, port = PORT, driver, config, onReport }) {
  const kit = readFileSync(join(import.meta.dirname, "kit.js"), "utf8");
  const pieces = new Map();
  return createServer(async (request, response) => {
    const url = new URL(request.url ?? "/", `http://localhost:${port}`);
    if (url.pathname === "/__report") {
      const kind = url.searchParams.get("kind");
      const id = url.searchParams.get("id");
      const i = Number(url.searchParams.get("i"));
      const n = Number(url.searchParams.get("n"));
      const list = pieces.get(id) ?? [];
      list[i] = url.searchParams.get("data") ?? "";
      pieces.set(id, list);
      if (list.filter((p) => p !== undefined).length === n) {
        pieces.delete(id);
        try {
          await onReport(kind, JSON.parse(list.join("")));
        } catch (error) {
          console.error(`host: a report of kind ${kind} failed: ${error && error.message ? error.message : error}`);
        }
      }
      response.writeHead(204).end();
      return;
    }
    if (url.pathname === "/__driver.js") {
      response.writeHead(200, { "content-type": MIME[".js"] }).end(driver);
      return;
    }
    if (url.pathname === "/__kit.js") {
      response.writeHead(200, { "content-type": MIME[".js"] }).end(kit);
      return;
    }
    let path = normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    if (path === "/" || path === "") path = "/index.html";
    const file = join(dist, path);
    if (!file.startsWith(dist) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end("Not found");
      return;
    }
    let body = readFileSync(file);
    if (path === "/index.html") {
      body = Buffer.from(body.toString("utf8").replace("</head>", `<script>window.__SMOKE__ = ${JSON.stringify(config)};</script><script src="/__kit.js"></script><script src="/__driver.js"></script></head>`));
    }
    response.writeHead(200, { "content-type": MIME[extname(file)] ?? "application/octet-stream" }).end(body);
  });
}

/** Starts `server` on `port`; rejects with a sentence that says what to do when the port is taken (the server can be tried again). */
export function listen(server, port = PORT) {
  return new Promise((resolveListen, reject) => {
    const failed = (error) => {
      server.off("listening", listening);
      reject(new Error(`port ${port} is not free (${error.message}); stop the vite dev server first`));
    };
    const listening = () => {
      server.off("error", failed);
      resolveListen();
    };
    server.once("error", failed);
    server.once("listening", listening);
    server.listen(port, "127.0.0.1");
  });
}

/**
 * Starts an Xvfb screen of `cssWidth` x `cssHeight` CSS pixels at device scale `scale` (the screen is `scale` times larger, and the app
 * runs with `GDK_SCALE=scale`). Xvfb picks a free display itself and says which on file descriptor 3. Resolves with `{ xvfb, display }`
 * (`display` like ":99"); rejects when Xvfb does not start.
 */
export async function startXvfb(scale, cssWidth = 1600, cssHeight = 1000) {
  const xvfb = spawn("Xvfb", ["-displayfd", "3", "-screen", "0", `${cssWidth * scale}x${cssHeight * scale}x24`, "-nolisten", "tcp"], { stdio: ["ignore", "ignore", "ignore", "pipe"] });
  const display = `:${await new Promise((resolveDisplay, reject) => {
    xvfb.stdio[3].once("data", (chunk) => resolveDisplay(String(chunk).trim()));
    xvfb.once("exit", () => reject(new Error("Xvfb did not start")));
  })}`;
  return { xvfb, display };
}

/**
 * Starts the debug app on `display` with its folder (`YELLOWMOON_HOME`) at `home`. `env` adds to (and overrides) the environment.
 * Resolves with `{ app, log }`; `log()` is what the app wrote to its output so far.
 */
export function spawnApp({ binary, display, home, scale = 1, env = {} }) {
  const app = spawn(binary, [], {
    env: { ...process.env, DISPLAY: display, YELLOWMOON_HOME: home, WEBKIT_DISABLE_COMPOSITING_MODE: "1", YELLOWMOON_LOG: "debug", NO_AT_BRIDGE: "1", ...(scale > 1 ? { GDK_SCALE: String(scale) } : {}), ...env },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  app.stdout.on("data", (d) => (log += d));
  app.stderr.on("data", (d) => (log += d));
  return { app, log: () => log };
}

/**
 * Takes a picture of the whole screen with ImageMagick's `import` (or of the `crop` of it, `{ x, y, width, height }` in device pixels) and
 * writes it to `file`. Returns whether it worked.
 */
export function captureScreen({ display, file, crop }) {
  const args = ["-display", display, "-window", "root"];
  if (crop) args.push("-crop", `${crop.width}x${crop.height}+${crop.x}+${crop.y}`, "+repage");
  args.push(file);
  return spawnSync("import", args, { stdio: "ignore" }).status === 0;
}
