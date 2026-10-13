#!/usr/bin/env node
// README screenshots of the REAL app showing real photos: the debug Tauri shell under Xvfb at a device scale of 2 (like a Retina display),
// driven the way a person uses it (drop the photos, open them, change tab, open Export), one PNG per state, the webview only (no window
// frame), shrunk with pngquant. See tools/readme/README.md.
//
//   npm run build && cargo build -p yellowmoon-app        (or pass --build)
//   npm run readme:screenshots -- --photos ~/Pictures/Lisbon --out /tmp/readme-shots
//
// The server, the Xvfb screen, the app and the screenshot are shared with the smoke test (`e2e/real/host.mjs`). The app starts clean: an
// empty `YELLOWMOON_HOME` (no imported looks, default preferences). Nothing is written next to the photos, and nothing is exported (the
// Export panel is only opened). Exit code 0 when every picture was taken, 1 when one could not be, 2 when the tool cannot run.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { PORT, ROOT, captureScreen, listen, serveFrontend, spawnApp, startXvfb } from "../../e2e/real/host.mjs";
import { UsageError, USAGE, chooseStates, cssColours, findPhotos, findTools, formatSize, halve, licenceProblem, parseArgs, parsePick, planDrop, pngSize, scaleRects, shrinkShot, spread } from "./lib.mjs";

const say = (text) => console.log(`readme-screenshots: ${text}`);
function cannot(reason, code = 2) {
  console.error(`readme-screenshots: ${reason}`);
  process.exit(code);
}

let options;
try {
  options = parseArgs(process.argv.slice(2));
} catch (error) {
  if (error instanceof UsageError) cannot(`${error.message}\n\n${USAGE}`);
  throw error;
}
if (options.help) {
  console.log(USAGE);
  process.exit(0);
}

const dist = resolve(options.dist ?? join(ROOT, "dist"));
const binary = resolve(options.binary ?? join(ROOT, "target/debug/yellowmoon-app"));
const out = resolve(options.out);
const scale = options.scale;
const cssWidth = 1440;
const cssHeight = 900;

if (process.platform !== "linux") cannot("it needs Linux with Xvfb (the app is the debug shell of the Linux build); the pictures are of that build, not of the Mac one");
for (const tool of ["Xvfb", "import"]) if (spawnSync("which", [tool], { stdio: "ignore" }).status !== 0) cannot(`${tool} is not installed (Xvfb and ImageMagick are needed)`);
let files;
try {
  files = findPhotos(options.photos);
} catch (error) {
  if (error instanceof UsageError) cannot(error.message);
  throw error;
}
const problem = licenceProblem(out, files, ROOT);
if (problem) cannot(problem);
let pick;
try {
  pick = parsePick(options.pick);
} catch (error) {
  if (error instanceof UsageError) cannot(error.message);
  throw error;
}

if (options.build) {
  say("building the frontend and the app (npm run build, cargo build -p yellowmoon-app)");
  for (const [command, args] of [["npm", ["run", "build"]], ["cargo", ["build", "-p", "yellowmoon-app"]]]) {
    const r = spawnSync(command, args, { cwd: ROOT, stdio: "inherit", env: { ...process.env, CARGO_INCREMENTAL: process.env.CARGO_INCREMENTAL ?? "0" } });
    if (r.status !== 0) cannot(`${command} ${args.join(" ")} failed`, 1);
  }
}
if (!existsSync(join(dist, "index.html"))) cannot(`${dist}/index.html is missing; run \`npm run build\` (or pass --build)`);
if (options.data && !existsSync(resolve(options.data))) cannot(`${options.data} does not exist`);
if (!existsSync(binary)) cannot(`${binary} is missing; run \`cargo build -p yellowmoon-app\` (or pass --build)`);

const states = chooseStates(options.states, files.length);
if (states.length === 0) cannot("no state to capture");
const tools = findTools();
if (options.optimise && !tools.pngquant) say(`pngquant is not installed (apt install pngquant, brew install pngquant): ${tools.magick ? "ImageMagick's own quantiser is used, which is worse" : "the pictures stay lossless and large"}`);
if (!tools.magick && options.oneX !== "never") say("ImageMagick's convert is missing: no 1x files");
mkdirSync(out, { recursive: true });

// What is next to the photos before the run, so that anything the app writes there (an export folder, a sidecar) is found after it.
const folders = [...new Set(files.map((f) => dirname(f)))];
const listing = () => folders.flatMap((d) => readdirSync(d).map((f) => join(d, f)));
const listedBefore = new Set(listing());

const work = mkdtempSync(join(tmpdir(), "ym-readme-"));
const home = join(work, "home");
const raw = join(work, "raw");
// The lossless captures are photographs you may not be allowed to publish: they go with the temporary folder, whichever way this ends.
process.on("exit", () => rmSync(work, { recursive: true, force: true }));
const config_dir = join(work, "config");
for (const dir of [home, raw, join(config_dir, "gtk-3.0")]) mkdirSync(dir, { recursive: true });
// Grey-scale text smoothing, not a colour-fringed one, whatever the machine's own settings are.
writeFileSync(join(config_dir, "gtk-3.0", "settings.ini"), "[Settings]\ngtk-xft-antialias=1\ngtk-xft-hinting=1\ngtk-xft-hintstyle=hintslight\ngtk-xft-rgba=none\n");
const drop = planDrop(options.photos.map((p) => resolve(p)), files, options.name, work);
for (const link of drop.links) {
  mkdirSync(dirname(link.to), { recursive: true });
  symlinkSync(link.from, link.to);
}

const config = {
  paths: drop.paths,
  states,
  pick,
  cssWidth,
  cssHeight,
  exportShown: options.exportTo ?? `~/Pictures/${drop.folder} export`,
  stableMs: options.stableMs,
  developTimeoutMs: 10 * 60 * 1000,
  frameTimeoutMs: 3 * 60 * 1000,
  settleTimeoutMs: 4 * 60 * 1000,
};
const driver = readFileSync(join(import.meta.dirname, "driver.js"), "utf8");

let display = "";
let finished = null;
const shots = [];
const failures = [];

/** A picture of the webview, from the screen, into `raw/` (checked: its size, and that the photo's place is not flat). */
function capture(request) {
  const file = join(raw, `${request.name}.png`);
  const w = request.window;
  // Without a window manager the window sits at the corner of the screen and has no frame; if it had one, the page would be smaller than
  // its window and the webview would start below the frame.
  const frameTop = Math.max(0, (w.outerHeight - w.innerHeight) * request.dpr);
  const crop = { x: Math.round(w.screenX * request.dpr), y: Math.round(w.screenY * request.dpr + frameTop), width: request.view.width, height: request.view.height };
  if (!captureScreen({ display, file, crop })) {
    failures.push(`${request.name}: ImageMagick's import failed`);
    return;
  }
  const size = pngSize(readFileSync(file));
  const expected = { width: cssWidth * scale, height: cssHeight * scale };
  const flaws = [];
  const warnings = [];
  if (!size || size.width !== crop.width || size.height !== crop.height) flaws.push(`captured ${size ? `${size.width}x${size.height}` : "nothing"}, wanted ${crop.width}x${crop.height}`);
  if (crop.width !== expected.width || crop.height !== expected.height) warnings.push(`the window is ${crop.width}x${crop.height} device pixels, not ${expected.width}x${expected.height} (${cssWidth}x${cssHeight} CSS pixels at ${scale}x)`);
  let photoSpread = null;
  if (request.photo && request.photo.width > 0 && tools.magick) {
    photoSpread = spread(file, request.photo, tools);
    if (photoSpread !== null && photoSpread < 0.02) flaws.push(`the photo's place is nearly flat (spread ${photoSpread.toFixed(3)}): no photo is drawn there`);
  }
  shots.push({ ...request, file, flaws, warnings, photoSpread, crop });
  say(`captured ${request.name}: ${request.what}${flaws.length ? ` -- PROBLEM: ${flaws.join("; ")}` : ""}${warnings.length ? ` -- note: ${warnings.join("; ")}` : ""}`);
  if (flaws.length) failures.push(`${request.name}: ${flaws.join("; ")}`);
}

/** Resizes the app's X window (the webview is smaller than the window by the menu bar). Needs xdotool; says so when it is missing. */
function fitWindow({ width, height }) {
  const ids = [];
  for (const search of [["search", "--pid", String(app.pid)], ["search", "--name", "^Yellowmoon$"]]) {
    const r = spawnSync("xdotool", search, { env: { ...process.env, DISPLAY: display }, encoding: "utf8" });
    if (r.error) {
      say("xdotool is not installed (apt install xdotool): the window keeps the size the app opens with, 25 CSS pixels short of the mockups' height");
      return;
    }
    ids.push(...String(r.stdout).split("\n").filter(Boolean));
    if (ids.length) break;
  }
  for (const id of ids) spawnSync("xdotool", ["windowsize", id, String(width), String(height)], { env: { ...process.env, DISPLAY: display } });
  say(`resized ${ids.length} window(s) to ${width}x${height} device pixels`);
}

const server = serveFrontend({
  dist,
  driver,
  config,
  onReport(kind, payload) {
    if (kind === "shot") {
      // A picture taken again (the screen changed under the first): the new one replaces the old.
      const at = shots.findIndex((s) => s.name === payload.name);
      if (at >= 0) shots.splice(at, 1);
      capture(payload);
    } else if (kind === "fit") fitWindow(payload);
    else if (kind === "note") say(`page: ${payload.text}`);
    else if (kind === "done") finished = payload;
  },
});
// The app loads its frontend from port 1420, which `npm run dev`, Playwright's server and the smoke test also use: wait for a run that is
// in the way (up to ten minutes), as the app can't be told another port.
for (let attempt = 0; ; attempt++) {
  try {
    await listen(server);
    break;
  } catch (error) {
    if (attempt >= 60) cannot(error.message);
    if (attempt === 0) say(`${error.message}; waiting for it to end`);
    await new Promise((r) => setTimeout(r, 10000));
  }
}
const screen = await startXvfb(scale).catch((error) => cannot(error.message));
display = screen.display;
let app = null;
const cleanup = () => {
  app?.kill("SIGKILL");
  screen.xvfb.kill("SIGKILL");
  try {
    server.close();
  } catch {
    // Already closed: nothing to do.
  }
};
process.on("exit", cleanup);
process.on("SIGINT", () => process.exit(130));
process.on("uncaughtException", (error) => {
  console.error(error);
  process.exit(2);
});
const spawned = spawnApp({ binary, display, home, scale, env: { XDG_CONFIG_HOME: config_dir, YELLOWMOON_LOG: "info", ...(options.data ? { YELLOWMOON_DATA: resolve(options.data) } : {}) } });
app = spawned.app;
say(`${files.length} photo${files.length === 1 ? "" : "s"} (${drop.folder}), states ${states.join(", ")}, device scale ${scale}, app folder ${home}`);

const started = Date.now();
while (finished === null && Date.now() - started < options.timeoutMin * 60 * 1000 && app.exitCode === null) await new Promise((r) => setTimeout(r, 500));
if (finished === null) {
  const reason = app.exitCode === null ? `timed out after ${options.timeoutMin} minutes` : `the app exited with ${app.exitCode}`;
  failures.push(reason);
  captureScreen({ display, file: join(raw, "failure.png") });
} else if (!finished.ok) {
  // An app older than the data it reads says so in a sentence about a file of data/ (the debug app reads the repository's data/).
  const stale = /unknown field|missing field|invalid type|registry\.json|profile\.json/.test(finished.error ?? "");
  failures.push(finished.error + (stale ? " (the app may be older than data/; build it again, or give --data a copy of data/ from git: see tools/readme/README.md)" : ""));
}
app.kill("SIGTERM");
await new Promise((r) => setTimeout(r, 800));
// The app, the screen and port 1420 are not needed for shrinking: free them for whoever else wants the port (shrinking takes minutes).
cleanup();

const added = listing().filter((f) => !listedBefore.has(f));
if (added.length) failures.push(`the app wrote next to the photos: ${added.join(", ")}`);

// --- Shrink the pictures and write them ------------------------------------------------------------------------------------------

// The flat colours of the interface (the design tokens): kept exact when a picture is cut down to 256 colours, however small the part of
// the picture that is drawn in them (the moon mark at half size).
const tokensFile = join(ROOT, "design/tokens.css");
const keepColours = existsSync(tokensFile) ? cssColours(readFileSync(tokensFile, "utf8")) : [];
if (keepColours.length === 0) say("design/tokens.css is missing: small flat colours (the moon mark at half size) may be lost when the pictures are shrunk");

const written = [];
for (const shot of shots) {
  const base = shot.name;
  const target = join(out, `${base}.png`);
  const result = options.optimise
    ? shrinkShot(shot.file, target, { maxBytes: options.maxKb * 1000, minColours: options.minColours, dither: options.dither, tools, keepColours, photoRects: shot.photoRects ?? [], gradientRects: shot.gradientRects ?? [], uiRects: shot.uiRects ?? [] })
    : (copyFileSync(shot.file, target), { bytes: statSync(target).size, colours: 0, underTarget: statSync(target).size <= options.maxKb * 1000, lossless: true });
  const entry = { file: target, state: shot.name, what: shot.what, bytes: result.bytes, colours: result.colours, lossless: result.lossless, underTarget: result.underTarget, width: shot.view.width, height: shot.view.height, photoSpread: shot.photoSpread };
  written.push(entry);
  if (options.oneX !== "never" && tools.magick && scale === 2) {
    const small = join(out, `${base}-1x.png`);
    const tmp = join(raw, `${base}-1x.raw.png`);
    halve(shot.file, tmp, tools);
    const half = { photoRects: scaleRects(shot.photoRects ?? [], 0.5), gradientRects: scaleRects(shot.gradientRects ?? [], 0.5), uiRects: scaleRects(shot.uiRects ?? [], 0.5) };
    const r1 = options.optimise ? shrinkShot(tmp, small, { maxBytes: options.maxKb * 1000, minColours: options.minColours, dither: options.dither, tools, keepColours, ...half }) : (copyFileSync(tmp, small), { bytes: statSync(small).size, colours: 0, lossless: true, underTarget: true });
    if (options.oneX === "always" || r1.bytes < result.bytes * 0.6) written.push({ file: small, state: shot.name, what: `${shot.what} (1x)`, bytes: r1.bytes, colours: r1.colours, lossless: r1.lossless, underTarget: r1.bytes <= options.maxKb * 1000, width: shot.view.width / 2, height: shot.view.height / 2 });
    else rmSync(small, { force: true });
  }
  if (options.keepRaw) {
    mkdirSync(join(out, "raw"), { recursive: true });
    copyFileSync(shot.file, join(out, "raw", `${base}.png`));
    // Where the photographs, the gradients and the interface drawn over them are, for shrinking the picture again later.
    writeFileSync(join(out, "raw", `${base}.json`), JSON.stringify({ photoRects: shot.photoRects, gradientRects: shot.gradientRects, uiRects: shot.uiRects }, null, 1) + "\n");
  }
}
writeFileSync(
  join(out, "manifest.json"),
  JSON.stringify(
    {
      generated_by: "tools/readme/screenshots.mjs",
      photos: files.map((f) => f.split("/").pop()),
      folder_shown: drop.folder,
      scale,
      window_css_px: [cssWidth, cssHeight],
      warnings: [...new Set(shots.flatMap((s) => s.warnings))],
      images: written.map((w) => ({ file: w.file.split("/").pop(), shows: w.what, bytes: w.bytes, width_px: w.width, height_px: w.height, colours: w.colours || null })),
    },
    null,
    2,
  ) + "\n",
);

console.log("");
console.log("file".padEnd(24) + "size".padStart(10) + "  px".padEnd(12) + "colours  shows");
for (const w of written) console.log(w.file.split("/").pop().padEnd(24) + formatSize(w.bytes).padStart(10) + `  ${w.width}x${w.height}`.padEnd(12) + String(w.lossless ? "lossless" : w.colours).padEnd(9) + w.what + (w.underTarget || !options.optimise ? "" : `  (over ${options.maxKb} KB)`));
console.log("");
const over = options.optimise ? written.filter((w) => !w.underTarget && !/\(1x\)/.test(w.what)) : [];
if (over.length) say(`${over.length} picture(s) are over ${options.maxKb} KB at the fewest colours allowed (${options.minColours}); raise --max-kb, lower --min-colours, or accept the size`);

const code = failures.length ? 1 : 0;
if (failures.length) console.error(`readme-screenshots: problems:\n  ${failures.join("\n  ")}`);
else say(`${shots.length} picture(s) written to ${out}`);
process.exit(code);
