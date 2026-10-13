#!/usr/bin/env node
// The real frontend in the real Tauri shell (M3-02, M3-03, M3-04 frontend side), on Linux under Xvfb. Playwright can only drive the mock; this
// is the one check that the built frontend and the Rust shell agree: the generated commands, the events, the fallback viewport's frames
// served on the `ym-frame` scheme, the thumbnails, Pick white through the core, an export, and the update cell's `export_busy`.
//
//   npm run build && cargo build -p yellowmoon-app
//   node e2e/real/shell-smoke.mjs [folder-with-ARW-files]
//
// How it works: the debug app loads its frontend from `devUrl` (http://localhost:1420). This script serves `dist/` there, with `kit.js` (the
// helpers) and `driver.js` added to index.html; the driver runs inside the webview, drives the app and reports each check back as an image
// request. The server, the Xvfb screen, the app and the screenshot are in `host.mjs`, which `tools/readme/screenshots.mjs` uses too. Nothing is
// written next to the photos: the app's folder (`YELLOWMOON_HOME`) and the export folder are temporary. The ARW files are only read, in
// place (the real ones of `testdata/real` are for local evaluation and are never copied).
//
// Needs Xvfb (and ImageMagick's `import` for the screenshots). Exit code 0 when every check passed, 1 when one failed, 2 when it cannot run.
//
// Environment: `YM_SMOKE_SCALE=2` runs the app at a device scale of 2, like a Retina display (a CSS pixel is then two device pixels, and the
// frame twice the cell); `YM_SMOKE_ONLY=geometry` runs only the failure-path folder with its Pick white checks, for a quick run;
// `YM_SMOKE_TIMEOUT_MS`, `YM_SMOKE_OUT` and `YM_SMOKE_DISPLAY` change the time budget, the report folder and the X display.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
// The server, the Xvfb screen, the app and the screenshot are shared with `tools/readme/screenshots.mjs`.
import { PORT, captureScreen, listen, serveFrontend, spawnApp, startXvfb } from "./host.mjs";

const root = resolve(import.meta.dirname, "../..");
const dist = join(root, "dist");
const binary = join(root, "target/debug/yellowmoon-app");
const folder = resolve(process.argv[2] ?? (existsSync(join(root, "testdata/real/files")) ? join(root, "testdata/real/files") : join(root, "testdata/synthetic/arw")));
const port = PORT;
let display = process.env.YM_SMOKE_DISPLAY ?? "";
const out = process.env.YM_SMOKE_OUT ?? join(root, "node_modules/.cache/yellowmoon-e2e/shell-smoke");
const budgetMs = Number(process.env.YM_SMOKE_TIMEOUT_MS ?? 15 * 60 * 1000);
// `YM_SMOKE_SCALE=2` runs the app at a device scale of 2 (GDK_SCALE, on an Xvfb screen twice as large), like a Retina display: the frame is
// then twice the cell's size in device pixels, and a CSS pixel is two device pixels, which 1 can't tell apart.
const scale = Math.max(1, Math.round(Number(process.env.YM_SMOKE_SCALE ?? 1)) || 1);

// The temporary folders of this run: a run that cannot start (the port is taken, Xvfb is missing) removes them, so that it leaves nothing behind.
const temporary = [];
function cannot(reason) {
  console.error(`shell-smoke: cannot run: ${reason}`);
  for (const dir of temporary) rmSync(dir, { recursive: true, force: true });
  process.exit(2);
}
if (process.platform !== "linux") cannot("it needs Linux with Xvfb");
if (!existsSync(join(dist, "index.html"))) cannot("dist/ is missing; run `npm run build`");
if (!existsSync(binary)) cannot("target/debug/yellowmoon-app is missing; run `cargo build -p yellowmoon-app`");
if (!existsSync(folder) || readdirSync(folder).filter((f) => /\.arw$/i.test(f)).length === 0) cannot(`${folder} has no ARW files`);
if (spawnSync("which", ["Xvfb"]).status !== 0) cannot("Xvfb is not installed");
mkdirSync(out, { recursive: true });

// What is next to the photos before the run, so that anything the app writes there (an export folder, a sidecar) is found after it.
const listing = () => [...readdirSync(folder).map((f) => `${folder}/${f}`), ...readdirSync(dirname(folder)).map((f) => `${dirname(folder)}/${f}`)];
const listedBefore = new Set(listing());

const home = mkdtempSync(join(tmpdir(), "ym-smoke-home-"));
const exportDir = mkdtempSync(join(tmpdir(), "ym-smoke-export-"));
temporary.push(home, exportDir);
const checks = [];
let finished = null;

// A second folder for the failure paths: two of the synthetic files (made by our own writer, so copying them is fine), a file that is
// not a raw file at all, and a text file. Never made from the real files.
const brokenDir = mkdtempSync(join(tmpdir(), "ym-smoke-broken-"));
temporary.push(brokenDir);
const synthetic = join(root, "testdata/synthetic");
const goodFiles = existsSync(synthetic) ? readdirSync(synthetic).filter((f) => /\.arw$/i.test(f)).slice(0, 2) : [];
for (const f of goodFiles) copyFileSync(join(synthetic, f), join(brokenDir, f));
writeFileSync(join(brokenDir, "DSC00000_broken.ARW"), "This is not a raw file.\n".repeat(200));
writeFileSync(join(brokenDir, "notes.txt"), "Not a photo.\n");
const config = {
  brokenDir: goodFiles.length === 2 ? brokenDir : null,
  only: process.env.YM_SMOKE_ONLY ?? "",
  dir: folder,
  exportDir,
  pickAt: [0.5, 0.5],
  developTimeoutMs: 8 * 60 * 1000,
  frameTimeoutMs: 3 * 60 * 1000,
  lookTimeoutMs: 3 * 60 * 1000,
  exportTimeoutMs: 8 * 60 * 1000,
};
const driver = readFileSync(join(import.meta.dirname, "driver.js"), "utf8");

function screenshot(name) {
  const file = join(out, `${name}.png`);
  console.log(captureScreen({ display, file }) ? `shell-smoke: screenshot ${file}` : "shell-smoke: no screenshot (ImageMagick `import` failed)");
}

const server = serveFrontend({
  dist,
  port,
  driver,
  config,
  onReport(kind, payload) {
    if (kind === "check") {
      checks.push(payload);
      console.log(`${payload.ok ? "  ok  " : " FAIL "} ${payload.name}${payload.detail ? `: ${payload.detail}` : ""}`);
    } else if (kind === "screenshot") screenshot(payload.name);
    else if (kind === "done") finished = payload;
  },
});
await listen(server, port).catch((error) => cannot(error.message));

// Xvfb picks a free display itself.
const screen = await startXvfb(scale).catch((error) => cannot(error.message));
const xvfb = screen.xvfb;
display = screen.display;
let app = null;
// However this script ends, nothing it started stays behind: a leftover app would go on reporting to the next run.
const cleanup = () => {
  app?.kill("SIGKILL");
  xvfb.kill("SIGKILL");
  server.close();
};
process.on("exit", cleanup);
process.on("uncaughtException", (error) => {
  console.error(error);
  process.exit(2);
});
const spawned = spawnApp({ binary, display, home, scale });
app = spawned.app;
const appOutput = spawned.log;
console.log(`shell-smoke: ${folder} (${readdirSync(folder).filter((f) => /\.arw$/i.test(f)).length} ARW files), app folder ${home}, export folder ${exportDir}, device scale ${scale}`);

const started = Date.now();
while (finished === null && Date.now() - started < budgetMs && app.exitCode === null) await new Promise((r) => setTimeout(r, 500));
if (finished === null) {
  checks.push({ name: "the smoke test finished", ok: false, detail: app.exitCode === null ? "timed out" : `the app exited with ${app.exitCode}` });
  screenshot("timeout");
}

const exported = readdirSync(exportDir);
if (config.only === "") {
  checks.push({ name: "the export wrote its files, and only there", ok: exported.length > 0, detail: exported.map((f) => `${f} (${statSync(join(exportDir, f)).size} bytes)`).join(", ") || "nothing" });
  console.log(`${exported.length > 0 ? "  ok  " : " FAIL "} the export folder holds: ${exported.join(", ") || "nothing"}`);
}
const added = listing().filter((f) => !listedBefore.has(f));
console.log(`${added.length === 0 ? "  ok  " : " FAIL "} nothing was written next to the photos${added.length ? `: ${added.join(", ")}` : ""}`);
checks.push({ name: "nothing was written next to the photos", ok: added.length === 0, detail: added.join(", ") });

app.kill("SIGTERM");
await new Promise((r) => setTimeout(r, 1000));
const logFile = join(home, "logs", "yellowmoon.log");
const fileLog = existsSync(logFile) ? readFileSync(logFile, "utf8") : "";
writeFileSync(join(out, "report.json"), JSON.stringify({ folder, checks, appLog: appOutput().slice(-20000), fileLog: fileLog.slice(-20000) }, null, 2));
const failed = checks.filter((c) => !c.ok);
console.log(`shell-smoke: ${checks.length - failed.length} of ${checks.length} checks passed; report in ${join(out, "report.json")}`);
if (failed.length > 0) console.log(`shell-smoke: last lines of the app's output and log:\n${appOutput().split("\n").slice(-15).join("\n")}\n${fileLog.split("\n").slice(-25).join("\n")}`);
// The temporary folders go with a passing run; after a failing one they stay, for a look at what the app wrote.
if (failed.length === 0) for (const dir of [home, exportDir, brokenDir]) rmSync(dir, { recursive: true, force: true });
else console.log(`shell-smoke: kept ${home}, ${exportDir} and ${brokenDir}`);
process.exit(failed.length > 0 ? 1 : 0);
