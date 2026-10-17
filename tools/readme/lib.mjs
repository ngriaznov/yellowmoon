// The parts of the README screenshot tool that need no app: reading the command line, finding the photos, the rules for where pictures may
// be written, and shrinking a PNG. Tested by `screenshots.test.mjs`; used by `screenshots.mjs`.
import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync } from "node:fs";
import { basename, dirname, extname, join, relative, resolve, sep } from "node:path";

/** Every state the tool can capture, in the order it captures them. `light` needs a second photo to be worth it. */
export const ALL_STATES = ["start", "look", "light", "colour", "colour-warp", "detail", "export"];
/** The PNG each state is written to (the name of the state is the name of the file). */
export const FILE_OF_STATE = { start: "start-ready", look: "photo-look", light: "photo-light", colour: "photo-colour", "colour-warp": "photo-colour-warp", detail: "photo-detail", export: "photo-export" };
const PICK_KEYS = ["look", "light", "colour", "colour-warp", "detail", "export"];

/** The most pins `--warp` may add (the card's own limit, `WARP_MAX_PINS`). */
export const WARP_MAX_PINS = 16;
/** How far `--warp` may move a pin on the chroma plane, either way on each axis (the plane's edge, `WARP_PLANE_EXTENT`). */
export const WARP_MOVE_LIMIT = 0.3;

/** The ways a capture can be reduced to an 8-bit PNG (`--palette`): one palette over the whole picture, or one per kind of region. */
export const PALETTES = ["whole", "regions"];

/**
 * What `--min-colours` and `--dither` are when they are not given, for each palette mode. `whole` has to keep the interface's gradients
 * smooth, which a floor of 96 colours (a photograph's) does not, and dithers fully (pngquant's own default); `regions` keeps the
 * interface flat and exact, so it dithers less and may cut the photographs further.
 */
export function paletteDefaults(palette) {
  return palette === "regions" ? { minColours: 96, dither: 0.5 } : { minColours: 192, dither: 1 };
}

export const USAGE = `Usage: npm run readme:screenshots -- --photos <folder-or-files...> --out <folder> [options]

Runs the real app (the debug Tauri shell, under Xvfb) on your photos and writes one PNG per state of the interface:
  start-ready   the Start screen, every photo developed
  photo-look    a photo open, Look tab
  photo-light   a second photo, Light tab
  photo-colour  the most colourful photo, Colour tab (the white balance card)
  photo-colour-warp  the same photo, Colour tab scrolled to the Colour warp card (with the pins of --warp, if any)
  photo-detail  a photo, Detail tab
  photo-export  the Export panel over a photo

Options
  --photos <path...>   a folder of ARW files, or ARW files (repeat the flag, or list several after it)   [required]
  --out <folder>       where the PNGs go; created if missing                                              [required]
  --name <text>        the folder name the app shows (default: the name of the photos' folder); the photos are
                       linked, not copied, into a temporary folder of that name
  --export-to <text>   the folder the Export panel says it saves to (default: "~/Pictures/<folder name> export", as in the mockup;
                       the real default would show your own paths). Nothing is exported.
  --pick <spec>        which photo each state shows. A value for all ("--pick DSC04204"), or per state
                       ("--pick look=first,light=2,colour=colourful,detail=DSC04204,export=last").
                       Values: first, second, last, colourful, a number (1 is the first in the filmstrip),
                       or a file name (or part of one). Default: first for look, detail and export; the second
                       photo for light; the most colourful for colour; colour-warp takes the photo of colour.
  --warp <pins>        pins for photo-colour-warp, added the way a person does (Pick colour, a click on the photo, then the
                       arrow keys): "x,y:da,db; x,y:da,db". x,y is the spot of the photo cell that is clicked, as a share of
                       its width and height (0 to 1); da,db is how far the pin is moved on the chroma plane (right and up,
                       at most ${WARP_MOVE_LIMIT} either way; the plane is 0.6 across). The pins are removed again after the picture.
                       Default: none (the card with the photo's colours on the plane).
  --states <list>      comma-separated, from: ${ALL_STATES.join(", ")} (default: all; light is left out for one photo)
  --scale <1|2>        device scale of the screen (default 2, like a Retina display; the window is 1440 x 900 CSS pixels)
  --palette <mode>     how a capture is cut down to an 8-bit PNG (default whole):
                         whole    one palette over the whole picture, dithered. Right for the glass interface (gradients, blur)
                         regions  one palette each for the interface (exact colours, no dithering), the colour wheels and
                                  histogram, and the photographs. Right for a flat-colour interface
  --max-kb <n>         size each PNG should stay under (default 600); the palette is cut until it does, down to --min-colours
  --min-colours <n>    the fewest colours the picture may be cut to (default 192 for whole, 96 for regions: a photo bands
                       visibly below about 96, and a gradient below about 192)
  --dither <0..1>      pngquant's dithering strength (default 1 for whole, 0.5 for regions: less is smaller and bands more)
  --one-x <mode>       also write a half-size NAME-1x.png, made from the lossless capture (Lanczos), not from the 2x PNG: auto (when it
                       is less than 60 % of the 2x file), always, never (default auto)
  --no-optimise        keep the captures as they are (lossless, large)
  --build              run \`npm run build\` and \`cargo build -p yellowmoon-app\` first
  --dist <folder>      the built frontend (default dist/)
  --binary <file>      the debug app (default target/debug/yellowmoon-app)
  --data <folder>      the data/ folder the app reads (YELLOWMOON_DATA); by default the repository's own. Point it at a copy
                       (\`git archive HEAD data | tar -x -C /tmp/snapshot\`) when the app is older than data/ or data/ is being edited
  --stable-ms <n>      how long the screen must stand still before a capture (default 1500)
  --timeout-min <n>    give up after this many minutes (default 20)
  --keep-raw           also keep the lossless captures, in <out>/raw
  -h, --help           this text
`;

/** An error in what the person typed: the runner prints its message and exits with 2. */
export class UsageError extends Error {}

/**
 * Reads the command line (without `node` and the script). Returns the options with their defaults (`minColours` and `dither` default to
 * what `paletteDefaults` says for `palette`). Throws `UsageError`.
 * `--photos` takes every following word up to the next flag, so `--photos a.ARW b.ARW --out x` works as well as `--photos a --photos b`.
 */
export function parseArgs(argv) {
  const o = { photos: [], out: null, name: null, exportTo: null, pick: "", states: null, scale: 2, palette: "whole", maxKb: 600, minColours: null, dither: null, oneX: "auto", optimise: true, build: false, dist: null, binary: null, data: null, stableMs: 1500, timeoutMin: 20, keepRaw: false, warp: [], help: false };
  const value = (flag, i) => {
    const v = argv[i + 1];
    if (v === undefined || (v.startsWith("--") && v.length > 2)) throw new UsageError(`${flag} needs a value`);
    return v;
  };
  const number = (flag, text, min) => {
    const n = Number(text);
    if (!Number.isFinite(n) || n < min) throw new UsageError(`${flag} needs a number of at least ${min}, not "${text}"`);
    return n;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const [flag, inline] = a.startsWith("--") && a.includes("=") ? [a.slice(0, a.indexOf("=")), a.slice(a.indexOf("=") + 1)] : [a, undefined];
    const next = () => {
      if (inline !== undefined) return inline;
      const v = value(flag, i);
      i++;
      return v;
    };
    switch (flag) {
      case "-h":
      case "--help":
        o.help = true;
        break;
      case "--photos": {
        if (inline !== undefined) o.photos.push(inline);
        else {
          let n = 0;
          while (argv[i + 1] !== undefined && !argv[i + 1].startsWith("-")) {
            o.photos.push(argv[++i]);
            n++;
          }
          if (n === 0) throw new UsageError("--photos needs a folder or ARW files");
        }
        break;
      }
      case "--out":
        o.out = next();
        break;
      case "--name":
        o.name = next();
        // The name becomes a folder of links in a temporary folder: "." or ".." would be the temporary folder or the one above it.
        if (!o.name.trim() || /^\.+$/.test(o.name.trim()) || /[/\\]/.test(o.name)) throw new UsageError(`--name is a folder name shown in the app's header, not "${o.name}"`);
        break;
      case "--export-to":
        o.exportTo = next();
        break;
      case "--pick":
        o.pick = next();
        break;
      case "--states":
        o.states = next().split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
        break;
      case "--warp":
        o.warp = parseWarp(next());
        break;
      case "--scale":
        o.scale = number(flag, next(), 1);
        if (![1, 2].includes(o.scale)) throw new UsageError("--scale is 1 or 2");
        break;
      case "--palette":
        o.palette = next();
        if (!PALETTES.includes(o.palette)) throw new UsageError("--palette is whole or regions");
        break;
      case "--max-kb":
        o.maxKb = number(flag, next(), 1);
        break;
      case "--min-colours":
      case "--min-colors":
        o.minColours = number(flag, next(), 2);
        break;
      case "--dither":
        o.dither = number(flag, next(), 0);
        if (o.dither > 1) throw new UsageError("--dither is between 0 and 1");
        break;
      case "--one-x":
        o.oneX = next();
        if (!["auto", "always", "never"].includes(o.oneX)) throw new UsageError("--one-x is auto, always or never");
        break;
      case "--no-optimise":
      case "--no-optimize":
        o.optimise = false;
        break;
      case "--build":
        o.build = true;
        break;
      case "--dist":
        o.dist = next();
        break;
      case "--data":
        o.data = next();
        break;
      case "--binary":
        o.binary = next();
        break;
      case "--stable-ms":
        o.stableMs = number(flag, next(), 0);
        break;
      case "--timeout-min":
        o.timeoutMin = number(flag, next(), 1);
        break;
      case "--keep-raw":
        o.keepRaw = true;
        break;
      default:
        // A bare word after the other options is a photo: `npm run readme:screenshots -- --out x photos/`.
        if (!a.startsWith("-")) o.photos.push(a);
        else throw new UsageError(`unknown option ${a}; see --help`);
    }
  }
  // What the palette mode decides when the person did not (whichever order the flags came in).
  const byPalette = paletteDefaults(o.palette);
  o.minColours ??= byPalette.minColours;
  o.dither ??= byPalette.dither;
  if (o.help) return o;
  if (o.photos.length === 0) throw new UsageError("--photos is required");
  if (!o.out) throw new UsageError("--out is required");
  if (o.states) {
    const bad = o.states.filter((s) => !ALL_STATES.includes(s));
    if (bad.length) throw new UsageError(`unknown state ${bad.join(", ")}; the states are ${ALL_STATES.join(", ")}`);
  }
  return o;
}

/**
 * Reads `--pick`: "DSC04204" (one photo for every state) or "look=first,colour=colourful,export=3". Returns `{ all?, look?, light?, ... }`.
 * Throws `UsageError` for a state that does not take a photo or an empty value.
 */
export function parsePick(text) {
  const pick = {};
  for (const part of String(text || "").split(",").map((s) => s.trim()).filter(Boolean)) {
    const at = part.indexOf("=");
    if (at < 0) {
      pick.all = part;
      continue;
    }
    let state = part.slice(0, at).trim().toLowerCase();
    const photo = part.slice(at + 1).trim();
    if (state === "color") state = "colour";
    if (state === "color-warp") state = "colour-warp";
    if (!PICK_KEYS.includes(state)) throw new UsageError(`--pick: ${state} takes no photo; the states with a photo are ${PICK_KEYS.join(", ")}`);
    if (!photo) throw new UsageError(`--pick: ${state}= needs a photo`);
    pick[state] = photo;
  }
  return pick;
}

/**
 * Reads `--warp`: "x,y:da,db; x,y:da,db", one pin each. Returns `[{ at: [x, y], move: [da, db] }]`. `x,y` is a spot of the photo cell as
 * a share of its width and height; `da,db` moves the pin on the chroma plane (`:da,db` may be left out: the pin is added and not moved).
 * Throws `UsageError`.
 */
export function parseWarp(text) {
  const pins = [];
  for (const part of String(text || "").split(";").map((s) => s.trim()).filter(Boolean)) {
    const [spot, move = "0,0", extra] = part.split(":").map((s) => s.trim());
    const pair = (words, what) => {
      const n = words.split(",").map((w) => (w.trim() === "" ? NaN : Number(w)));
      if (n.length !== 2 || !n.every(Number.isFinite)) throw new UsageError(`--warp: ${what} is two numbers, not "${words}" (in "${part}")`);
      return n;
    };
    if (extra !== undefined) throw new UsageError(`--warp: one colon per pin, between the spot and the move (in "${part}")`);
    const at = pair(spot, "the spot x,y");
    const by = pair(move, "the move da,db");
    if (at.some((v) => v < 0 || v > 1)) throw new UsageError(`--warp: the spot is a share of the photo cell, from 0 to 1 (in "${part}")`);
    if (by.some((v) => Math.abs(v) > WARP_MOVE_LIMIT)) throw new UsageError(`--warp: a move is at most ${WARP_MOVE_LIMIT} either way (in "${part}")`);
    pins.push({ at, move: by });
  }
  if (pins.length > WARP_MAX_PINS) throw new UsageError(`--warp: at most ${WARP_MAX_PINS} pins`);
  return pins;
}

/** The states to capture, in order: `requested` (or all), without `light` for a single photo unless it was asked for by name. */
export function chooseStates(requested, photoCount) {
  if (requested) return ALL_STATES.filter((s) => requested.includes(s));
  return ALL_STATES.filter((s) => s !== "light" || photoCount > 1);
}

/** Lists the ARW files of what was given: a folder (one level deep, like the app) or ARW files. Throws `UsageError` for anything else. */
export function findPhotos(paths) {
  const files = [];
  for (const given of paths) {
    const path = resolve(given);
    if (!existsSync(path)) throw new UsageError(`${given} does not exist`);
    if (statSync(path).isDirectory()) {
      const found = readdirSync(path).filter((f) => /\.arw$/i.test(f) && !f.startsWith(".")).sort();
      if (found.length === 0) throw new UsageError(`${given} has no ARW files`);
      files.push(...found.map((f) => join(path, f)));
    } else if (/\.arw$/i.test(path)) files.push(path);
    else throw new UsageError(`${given} is not an ARW file or a folder`);
  }
  return [...new Set(files)];
}

/**
 * Decides what to drop on the app's window. A single folder is dropped as it is (the app names the session after it); anything else (files,
 * several folders, or a `name` to show) goes through a temporary folder of links, which the caller makes: `{ link: [{ from, to }...] }`.
 */
export function planDrop(given, files, name, tmp) {
  const folders = given.filter((g) => existsSync(g) && statSync(g).isDirectory()).map((g) => resolve(g));
  if (!name && folders.length === 1 && given.length === 1) return { paths: [folders[0]], links: [], folder: basename(folders[0]) };
  const shown = name || (folders.length === 1 ? basename(folders[0]) : "Photos");
  const dir = join(tmp, shown.replace(/[/\\]/g, "_"));
  return { paths: [dir], links: files.map((from) => ({ from, to: join(dir, basename(from)) })), folder: shown };
}

/**
 * `path` with every symbolic link in it resolved, also when the end of it does not exist yet (the nearest folder that exists is resolved
 * and the rest put back). The licence rule compares real places: a link to a file of testdata/ is that file, and so is an output folder
 * that is a link into the repository.
 */
export function realPlace(path) {
  const rest = [];
  let here = resolve(path);
  for (;;) {
    try {
      return join(realpathSync(here), ...rest.reverse());
    } catch {
      const up = dirname(here);
      if (up === here) return resolve(path);
      rest.push(basename(here));
      here = up;
    }
  }
}

/**
 * The licence rule: the nine files of testdata/real are for local evaluation only, so no picture made from them may be written inside the
 * repository. Returns a sentence when `out` is inside `root` and a photo is one of those (or lives under `testdata/`), else null. Links
 * are followed on both sides (`realPlace`).
 */
export function licenceProblem(out, files, root) {
  const inside = (child, parent) => {
    const r = relative(parent, child);
    return r === "" || (!r.startsWith("..") && !r.startsWith(sep) && !/^[a-zA-Z]:/.test(r));
  };
  const repository = realPlace(root);
  if (!inside(realPlace(out), repository)) return null;
  const local = files.find((f) => inside(realPlace(f), join(repository, "testdata")));
  if (!local) return null;
  return `${out} is inside the repository and ${basename(local)} comes from testdata/, whose files are for local evaluation only (unclear licences). Write the pictures outside the repository, or use photos you may publish.`;
}

// --- PNG files --------------------------------------------------------------------------------------------------------------------

/** "532 KB" or "1.8 MB". */
export function formatSize(bytes) {
  return bytes >= 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.round(bytes / 1000)} KB`;
}

/** The first of `names` found on the PATH, or null. */
export function which(...names) {
  for (const name of names) if (spawnSync("which", [name], { stdio: "ignore" }).status === 0) return name;
  return null;
}

/** The tools that can shrink a PNG here: pngquant (colour reduction) and oxipng or optipng (lossless recompression), else ImageMagick. */
export function findTools() {
  return { pngquant: which("pngquant"), lossless: which("oxipng", "optipng"), magick: which("magick", "convert") };
}

/** The width and height of a PNG (read from its header), or null. */
export function pngSize(bytes) {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  if (bytes.length < 24 || signature.some((b, i) => bytes[i] !== b)) return null;
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

function run(command, args) {
  const r = spawnSync(command, args, { stdio: ["ignore", "ignore", "pipe"] });
  if (r.status !== 0) throw new Error(`${command} failed (${r.status}): ${String(r.stderr).slice(0, 300)}`);
}

/** Recompresses a PNG without changing a pixel, with oxipng or optipng (`level` 1 to 5: how hard); a no-op when neither is installed. */
export function recompress(file, tools, level = 5) {
  if (tools.lossless === "oxipng") run("oxipng", ["-o", String(Math.min(level, 6)), "--strip", "all", "-q", file]);
  else if (tools.lossless === "optipng") run("optipng", [`-o${Math.min(level, 7)}`, "-strip", "all", "-quiet", file]);
}

/**
 * Writes a smaller PNG of `input` to `output`, the lossy way a screenshot allows: one palette over the whole picture, dithered. With
 * pngquant the palette is cut from 256 colours down, by bisection, to the largest palette whose file is under `maxBytes`, but never below
 * `minColours`; without pngquant ImageMagick's quantiser does the same. Returns `{ bytes, colours, underTarget, lossless }`
 * (`underTarget` is false when even `minColours` is over `maxBytes`: the file is then the one with `minColours`). With `lossless` (or no
 * tool at all) only the compression is improved. `dither` is pngquant's Floyd-Steinberg strength (0 to 1): less of it is smaller and bands
 * a little more; 0.5 keeps a sky smooth at 128 colours, and 1 (pngquant's own default) keeps a gradient smooth. `speed` is pngquant's
 * (1 is the slowest and best: on a 2880 x 1800 capture it took the same 2.5 s as 3 and came out 1 dB closer to the original).
 */
export function shrinkPng(input, output, { maxBytes, minColours = 96, tools, lossless = false, dither = 0.5, speed = 3 }) {
  if (lossless || (!tools.pngquant && !tools.magick)) {
    copyFileSync(input, output);
    recompress(output, tools);
    const bytes = statSync(output).size;
    return { bytes, colours: 0, underTarget: bytes <= maxBytes, lossless: true };
  }
  const tries = new Map();
  const file = (colours) => `${output}.try${colours}.png`;
  const attempt = (colours) => {
    if (!tries.has(colours)) {
      if (tools.pngquant) run("pngquant", ["--force", "--speed", String(speed), "--strip", `--floyd=${dither}`, "--quality", "0-100", "--output", file(colours), String(colours), input]);
      else run(tools.magick, [input, "-colors", String(colours), "-dither", "FloydSteinberg", "-strip", "-define", "png:compression-level=9", file(colours)]);
      tries.set(colours, statSync(file(colours)).size);
    }
    return tries.get(colours);
  };
  const top = 256;
  const bottom = Math.max(2, Math.min(top, Math.round(minColours)));
  let chosen = top;
  if (attempt(top) > maxBytes) {
    if (attempt(bottom) > maxBytes) chosen = bottom;
    else {
      // The file at `lo` fits, the one at `hi` does not: the largest palette that fits is between them.
      let lo = bottom;
      let hi = top;
      while (hi - lo > 8) {
        const mid = Math.round((lo + hi) / 2);
        if (attempt(mid) <= maxBytes) lo = mid;
        else hi = mid;
      }
      chosen = lo;
    }
  }
  // The step that is kept gets the lossless pass; the others go.
  recompress(file(chosen), tools, 5);
  const bytes = statSync(file(chosen)).size;
  rmSync(output, { force: true });
  renameSync(file(chosen), output);
  for (const colours of tries.keys()) if (colours !== chosen) rmSync(file(colours), { force: true });
  return { bytes, colours: chosen, underTarget: bytes <= maxBytes, lossless: false };
}

/**
 * Every colour of a picture with its pixel count, most frequent first: `[{ count, rgb: [r, g, b] }, ...]`. Needs ImageMagick.
 * (`histogram:info:` lists every colour of the picture with its count.)
 */
export function colourHistogram(file, tools) {
  const r = spawnSync(tools.magick, [file, "-depth", "8", "-format", "%c", "histogram:info:-"], { encoding: "utf8", maxBuffer: 1 << 30 });
  if (r.status !== 0) throw new Error(`${tools.magick} failed: ${String(r.stderr).slice(0, 300)}`);
  const colours = [];
  for (const line of String(r.stdout).split("\n")) {
    const m = /^\s*(\d+):\s*\(\s*(\d+),\s*(\d+),\s*(\d+)/.exec(line);
    if (m) colours.push({ count: Number(m[1]), rgb: [Number(m[2]), Number(m[3]), Number(m[4])] });
  }
  return colours.sort((a, b) => b.count - a.count);
}

/** The colours of a picture that count most, by pixel count: `[[r, g, b], ...]`, at most `n`. Needs ImageMagick. */
export function topColours(file, n, tools) {
  return colourHistogram(file, tools).slice(0, n).map((c) => c.rgb);
}

/**
 * The `#rrggbb` colours written in a piece of CSS (the design tokens), each once, as `[r, g, b]`. These are the flat colours of the
 * interface: `shrinkShot` keeps them exact however few pixels of them are on the picture.
 */
export function cssColours(css) {
  const seen = new Set();
  const colours = [];
  for (const m of String(css).matchAll(/#([0-9a-fA-F]{6})\b/g)) {
    const key = m[1].toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    colours.push([0, 2, 4].map((i) => parseInt(key.slice(i, i + 2), 16)));
  }
  return colours;
}

const hex = ([r, g, b]) => `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;

/** Scales rectangles (`{ x, y, width, height }`, device pixels) by `factor`, rounding outwards. */
export const scaleRects = (rects, factor) =>
  rects.map((r) => {
    const x = Math.floor(r.x * factor);
    const y = Math.floor(r.y * factor);
    return { x, y, width: Math.ceil((r.x + r.width) * factor) - x, height: Math.ceil((r.y + r.height) * factor) - y };
  });

/**
 * Like `shrinkPng`, for a screenshot, with two ways to make the 8-bit PNG (`palette`).
 *
 * `"whole"` (the default) is `shrinkPng` with the settings that suit the dark glass interface: one palette over the whole picture, dithered
 * (`paletteDefaults("whole")`: `dither` 1, `minColours` 192) and made at pngquant's slowest, best speed. The interface is blurred translucent
 * panes over a gradient, not flat colour, so it has thousands of shades. With the three palettes of `"regions"` the right-hand panel's
 * gradient was mapped to the 48 most frequent exact colours without dithering (blotchy, with dark teal patches) and the Light tab's cyan
 * Auto pill came out white, because the accent colours are not in the design tokens. One palette over everything, with dithering, kept both.
 *
 * `"regions"` is for a flat-colour interface, where one palette over everything loses what is small (the yellow moon beside a cyan title
 * block came out as mud, the colour wheels as grey discs), because a photo's pixels outnumber them a thousand to one. The three kinds of
 * thing on the picture get a palette each:
 *
 * - The interface (everything not listed below) keeps its `uiColours` most frequent exact colours, mapped without dithering, and every
 *   one of `keepColours` (the design tokens: `[[r, g, b], ...]`) that is on the picture at all. A small flat colour is not among the
 *   most frequent once the picture has been scaled down (the moon mark is 20 pixels at half size, behind a thousand shades of grey from
 *   the text and the ringing of the resize), and without this it came out grey or as the colour of the title block beside it.
 * - `gradientRects` (the colour wheels, the histogram, the curve plot) get their own palette of `gradientColours`, dithered.
 * - `photoRects` (the photo cell, the filmstrip, the look tiles) get pngquant's palette of what is left of 256, searched as in `shrinkPng`
 *   for the largest that fits `maxBytes` (not below `minColours`).
 *
 * `uiRects` are overlays inside the photo rectangles that are interface after all (the DEFAULT tag, the look chip). The three are joined
 * (at most 256 colours) and written as an 8-bit PNG. Needs ImageMagick, pngquant and the rectangles, else `shrinkPng` (one palette)
 * is used.
 *
 * `minColours` and `dither` default to `paletteDefaults(palette)`.
 */
export function shrinkShot(input, output, { palette = "whole", maxBytes, minColours, dither, tools, photoRects = [], gradientRects = [], uiRects = [], uiColours = 48, gradientColours = 40, keepColours = [] }) {
  if (!PALETTES.includes(palette)) throw new Error(`palette is whole or regions, not "${palette}"`);
  minColours ??= paletteDefaults(palette).minColours;
  dither ??= paletteDefaults(palette).dither;
  if (palette === "whole") return shrinkPng(input, output, { maxBytes, minColours, dither, tools, speed: 1 });
  if (!tools.magick || !tools.pngquant || photoRects.length === 0) return shrinkPng(input, output, { maxBytes, minColours, dither, tools });
  const size = pngSize(readFileSync(input));
  if (!size) throw new Error(`${input} is not a PNG`);
  const work = `${output}.work`;
  rmSync(work, { recursive: true, force: true });
  mkdirSync(work, { recursive: true });
  try {
    const file = (name) => join(work, name);
    const rect = (r) => `rectangle ${r.x},${r.y} ${r.x + r.width - 1},${r.y + r.height - 1}`;
    const mask = (name, white, black = []) =>
      run(tools.magick, ["-size", `${size.width}x${size.height}`, "xc:black", "-stroke", "none", "-fill", "white", ...white.flatMap((r) => ["-draw", rect(r)]), "-fill", "black", ...black.flatMap((r) => ["-draw", rect(r)]), "-type", "Grayscale", file(name)]);
    // A mask is white where its group is drawn. The picture with only that group left (the rest white) goes to the quantiser.
    const only = (maskName, outName) => run(tools.magick, [input, "(", file(maskName), "-negate", ")", "-compose", "Screen", "-composite", file(outName)]);
    mask("photos-mask.png", photoRects, uiRects);
    mask("gradients-mask.png", gradientRects);
    mask("any-mask.png", [...photoRects, ...gradientRects], uiRects);
    only("photos-mask.png", "photos.png");
    if (gradientRects.length) only("gradients-mask.png", "gradients.png");
    // The interface: the picture with the other two groups blacked out, its most frequent colours exact.
    run(tools.magick, [input, "(", file("any-mask.png"), "-negate", ")", "-compose", "Multiply", "-composite", file("interface.png")]);
    const histogram = colourHistogram(file("interface.png"), tools);
    const present = new Map(histogram.map((c) => [hex(c.rgb), c.rgb]));
    const palette = [];
    const inPalette = new Set();
    // The colours to keep exactly first (those that are on the picture), then the most frequent ones.
    for (const colour of [...keepColours.filter((c) => present.has(hex(c))), ...histogram.slice(0, uiColours).map((c) => c.rgb)]) {
      if (inPalette.has(hex(colour))) continue;
      inPalette.add(hex(colour));
      palette.push(colour);
    }
    run(tools.magick, ["-size", "1x1", ...palette.map((c) => `xc:${hex(c)}`), "+append", file("palette.png")]);
    run(tools.magick, [input, "+dither", "-remap", file("palette.png"), file("ui.png")]);
    // The gradients, with a palette of their own, laid over the interface.
    let base = file("ui.png");
    let spent = palette.length;
    if (gradientRects.length) {
      run("pngquant", ["--force", "--speed", "3", "--strip", `--floyd=${dither}`, "--quality", "0-100", "--output", file("gradients-q.png"), String(gradientColours), file("gradients.png")]);
      run(tools.magick, [file("ui.png"), file("gradients-q.png"), file("gradients-mask.png"), "-composite", file("ui-gradients.png")]);
      base = file("ui-gradients.png");
      spent += gradientColours;
    }

    const room = 256 - spent;
    const bottom = Math.max(2, Math.min(room, Math.round(minColours)));
    const tries = new Map();
    const attempt = (colours) => {
      if (!tries.has(colours)) {
        const quantised = file(`q${colours}.png`);
        run("pngquant", ["--force", "--speed", "3", "--strip", `--floyd=${dither}`, "--quality", "0-100", "--output", quantised, String(colours), file("photos.png")]);
        const joined = file(`j${colours}.png`);
        run(tools.magick, [base, quantised, file("photos-mask.png"), "-composite", joined]);
        const final = file(`f${colours}.png`);
        // At most 256 colours now, so this only writes the palette (no dithering, nothing to lose).
        run("pngquant", ["--force", "--nofs", "--speed", "3", "--strip", "--quality", "0-100", "--output", final, "256", joined]);
        tries.set(colours, statSync(final).size);
        rmSync(quantised, { force: true });
        rmSync(joined, { force: true });
      }
      return tries.get(colours);
    };
    let chosen = room;
    if (attempt(room) > maxBytes) {
      if (attempt(bottom) > maxBytes) chosen = bottom;
      else {
        let lo = bottom;
        let hi = room;
        while (hi - lo > 8) {
          const mid = Math.round((lo + hi) / 2);
          if (attempt(mid) <= maxBytes) lo = mid;
          else hi = mid;
        }
        chosen = lo;
      }
    }
    recompress(file(`f${chosen}.png`), tools, 5);
    const bytes = statSync(file(`f${chosen}.png`)).size;
    rmSync(output, { force: true });
    renameSync(file(`f${chosen}.png`), output);
    return { bytes, colours: chosen + spent, photoColours: chosen, gradientColours: gradientRects.length ? gradientColours : 0, interfaceColours: palette.length, underTarget: bytes <= maxBytes, lossless: false };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

/** Makes `output` a copy of `input` at half its width and height (Lanczos), for the 1x file. Needs ImageMagick. */
export function halve(input, output, tools) {
  if (!tools.magick) throw new Error("halving a picture needs ImageMagick");
  run(tools.magick, [input, "-filter", "Lanczos", "-resize", "50%", "-strip", output]);
}

/** The standard deviation of the grey levels (0 to 1) of a rectangle of a picture: near 0 means flat (no photo drawn there). */
export function spread(file, rect, tools) {
  const r = spawnSync(tools.magick, [file, "-crop", `${rect.width}x${rect.height}+${rect.x}+${rect.y}`, "+repage", "-colorspace", "Gray", "-format", "%[fx:standard_deviation]", "info:"], { encoding: "utf8" });
  const n = Number(String(r.stdout).trim());
  return Number.isFinite(n) ? n : null;
}

/** Makes a folder (and its parents) if it is missing. */
export function ensureDir(dir) {
  mkdirSync(dir, { recursive: true });
}

/** The extension-less name of a file. */
export const stem = (file) => basename(file, extname(file));
