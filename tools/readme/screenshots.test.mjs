// Tests of the README screenshot tool's parts that need no app: the command line, the photos, the licence rule, shrinking a PNG, and the
// shared server of e2e/real/host.mjs. (The app itself is exercised by running the tool: see tools/readme/README.md.)
import { spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, statSync, symlinkSync, writeFileSync, readFileSync } from "node:fs";
import { get } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { listen, serveFrontend } from "../../e2e/real/host.mjs";
import { ALL_STATES, UsageError, chooseStates, cssColours, findPhotos, findTools, formatSize, halve, licenceProblem, parseArgs, parsePick, planDrop, pngSize, realPlace, scaleRects, shrinkPng, shrinkShot, spread, topColours } from "./lib.mjs";

let work;
beforeAll(() => {
  work = mkdtempSync(join(tmpdir(), "ym-readme-test-"));
});
afterAll(() => rmSync(work, { recursive: true, force: true }));

describe("parseArgs", () => {
  it("reads photos up to the next flag, and the defaults", () => {
    const o = parseArgs(["--photos", "a.ARW", "b.ARW", "--out", "shots"]);
    expect(o.photos).toEqual(["a.ARW", "b.ARW"]);
    expect(o).toMatchObject({ out: "shots", scale: 2, maxKb: 600, minColours: 96, dither: 0.5, oneX: "auto", optimise: true, build: false });
  });

  it("takes --photos more than once, with = and bare words", () => {
    const o = parseArgs(["--photos=a", "--out=x", "--photos", "b", "c"]);
    expect(o.photos).toEqual(["a", "b", "c"]);
    expect(o.out).toBe("x");
  });

  it("needs photos and an output folder", () => {
    expect(() => parseArgs(["--out", "x"])).toThrow(UsageError);
    expect(() => parseArgs(["--photos", "a"])).toThrow(/--out is required/);
    expect(() => parseArgs(["--photos"])).toThrow(/needs a folder/);
  });

  it("refuses what it does not know", () => {
    expect(() => parseArgs(["--photos", "a", "--out", "x", "--frobnicate"])).toThrow(/unknown option/);
    expect(() => parseArgs(["--photos", "a", "--out", "x", "--states", "look,nope"])).toThrow(/unknown state nope/);
    expect(() => parseArgs(["--photos", "a", "--out", "x", "--scale", "3"])).toThrow(/1 or 2/);
    expect(() => parseArgs(["--photos", "a", "--out", "x", "--one-x", "maybe"])).toThrow(/auto, always or never/);
    expect(() => parseArgs(["--photos", "a", "--out", "x", "--max-kb", "lots"])).toThrow(/needs a number/);
  });

  it("takes the options that change the output", () => {
    const o = parseArgs(["--photos", "a", "--out", "x", "--name", "Lisbon", "--states", "look,export", "--scale", "1", "--max-kb", "300", "--no-optimise", "--one-x", "never", "--build", "--keep-raw", "--export-to", "~/Pictures/Trip"]);
    expect(o).toMatchObject({ name: "Lisbon", states: ["look", "export"], scale: 1, maxKb: 300, optimise: false, oneX: "never", build: true, keepRaw: true, exportTo: "~/Pictures/Trip" });
  });

  it("takes a folder name for --name, not a path that would leave the temporary folder", () => {
    expect(parseArgs(["--photos", "a", "--out", "x", "--name", "Lisbon 2026"]).name).toBe("Lisbon 2026");
    for (const bad of ["..", ".", " ", "a/b", "..\\x"]) expect(() => parseArgs(["--photos", "a", "--out", "x", "--name", bad])).toThrow(/folder name/);
  });

  it("answers --help without the other options", () => {
    expect(parseArgs(["--help"]).help).toBe(true);
  });
});

describe("parsePick and chooseStates", () => {
  it("reads one photo for every state, or one per state", () => {
    expect(parsePick("DSC04204")).toEqual({ all: "DSC04204" });
    expect(parsePick("look=first, colour=colourful ,export=3")).toEqual({ look: "first", colour: "colourful", export: "3" });
    expect(parsePick("color=2")).toEqual({ colour: "2" });
    expect(parsePick("")).toEqual({});
  });

  it("rejects a state that takes no photo", () => {
    expect(() => parsePick("start=1")).toThrow(/takes no photo/);
    expect(() => parsePick("look=")).toThrow(/needs a photo/);
  });

  it("leaves out the Light tab for a single photo, unless it is asked for", () => {
    expect(chooseStates(null, 1)).toEqual(ALL_STATES.filter((s) => s !== "light"));
    expect(chooseStates(null, 3)).toEqual(ALL_STATES);
    expect(chooseStates(["export", "light"], 1)).toEqual(["light", "export"]);
  });
});

describe("photos and the drop", () => {
  it("lists the ARW files of a folder (one level, any case, sorted) and takes single files", () => {
    const folder = join(work, "Lisbon");
    mkdirSync(join(folder, "sub"), { recursive: true });
    for (const f of ["DSC2.ARW", "DSC1.arw", "notes.txt", "sub/DSC3.ARW", ".hidden.ARW"]) writeFileSync(join(folder, f), "x");
    expect(findPhotos([folder]).map((f) => f.split("/").pop())).toEqual(["DSC1.arw", "DSC2.ARW"]);
    expect(findPhotos([join(folder, "DSC2.ARW"), folder]).length).toBe(2);
    expect(() => findPhotos([join(folder, "notes.txt")])).toThrow(/not an ARW file/);
    expect(() => findPhotos([join(folder, "missing")])).toThrow(/does not exist/);
    expect(() => findPhotos([join(folder, "sub", "..", "sub")])).not.toThrow();
  });

  it("drops a single folder as it is, and links anything else into a folder of the shown name", () => {
    const folder = join(work, "Trip");
    mkdirSync(folder, { recursive: true });
    writeFileSync(join(folder, "A.ARW"), "x");
    const files = findPhotos([folder]);
    expect(planDrop([folder], files, null, work)).toEqual({ paths: [folder], links: [], folder: "Trip" });
    const named = planDrop([folder], files, "Lisbon", work);
    expect(named.folder).toBe("Lisbon");
    expect(named.paths).toEqual([join(work, "Lisbon")]);
    expect(named.links).toEqual([{ from: files[0], to: join(work, "Lisbon", "A.ARW") }]);
    expect(planDrop([files[0]], files, null, work).folder).toBe("Photos");
  });
});

describe("the licence rule", () => {
  const root = "/repo";
  it("refuses pictures of testdata/ inside the repository", () => {
    expect(licenceProblem("/repo/docs/screenshots", ["/repo/testdata/real/files/A.ARW"], root)).toMatch(/local evaluation only/);
    expect(licenceProblem("/repo", ["/repo/testdata/real/files/A.ARW"], root)).toMatch(/inside the repository/);
  });
  it("allows them outside the repository, and any photo that is not testdata", () => {
    expect(licenceProblem("/tmp/shots", ["/repo/testdata/real/files/A.ARW"], root)).toBeNull();
    expect(licenceProblem("/repo/docs/screenshots", ["/home/me/Pictures/A.ARW"], root)).toBeNull();
    expect(licenceProblem("/repo-other/shots", ["/repo/testdata/real/files/A.ARW"], root)).toBeNull();
  });
});

describe("the licence rule follows links", () => {
  it("sees a link to a file of testdata/ for what it is, and an output folder that is a link into the repository", () => {
    const repo = join(work, "repo");
    mkdirSync(join(repo, "testdata", "real"), { recursive: true });
    mkdirSync(join(repo, "docs"), { recursive: true });
    const real = join(repo, "testdata", "real", "A.ARW");
    writeFileSync(real, "x");
    const outside = join(work, "pictures");
    mkdirSync(outside, { recursive: true });
    const photoLink = join(outside, "Mine.ARW");
    symlinkSync(real, photoLink);
    const folderLink = join(work, "shots-link");
    symlinkSync(join(repo, "docs"), folderLink);
    // A link outside the repository to a file inside testdata/ is still a file of testdata/.
    expect(licenceProblem(join(repo, "docs", "screenshots"), [photoLink], repo)).toMatch(/local evaluation only/);
    // An output folder that is a link into the repository is inside it, also when the folder below it does not exist yet.
    expect(licenceProblem(join(folderLink, "screenshots"), [real], repo)).toMatch(/inside the repository/);
    // Neither: a real photo, an output folder that is really outside.
    expect(licenceProblem(join(folderLink, "screenshots"), [join(work, "Other.ARW")], repo)).toBeNull();
    expect(licenceProblem(join(outside, "shots"), [photoLink], repo)).toBeNull();
    expect(realPlace(join(folderLink, "a", "b"))).toBe(join(realPlace(repo), "docs", "a", "b"));
  });
});

describe("formatSize", () => {
  it("says KB below a megabyte and MB above", () => {
    expect(formatSize(532_400)).toBe("532 KB");
    expect(formatSize(1_840_000)).toBe("1.8 MB");
  });
});

describe("shrinking a PNG", () => {
  const tools = findTools();
  const can = !!tools.magick;
  let photo;
  beforeAll(() => {
    if (!can) return;
    photo = join(work, "plasma.png");
    // A photograph-like picture with a flat part, the way a screenshot is: noisy plasma in one half, white in the other.
    const r = spawnSync(tools.magick, ["-size", "320x200", "-seed", "7", "plasma:fractal", "(", "-size", "320x200", "xc:white", ")", "+append", "-crop", "320x200+160+0", "+repage", photo]);
    if (r.status !== 0) throw new Error("ImageMagick could not make a test picture: " + r.stderr);
  });

  it.skipIf(!can)("reads a PNG's size from its header", () => {
    expect(pngSize(readFileSync(photo))).toEqual({ width: 320, height: 200 });
    expect(pngSize(Buffer.from("not a png at all, really not a png"))).toBeNull();
  });

  it.skipIf(!can)("writes a smaller picture of the same size, with fewer colours as the target gets tighter", () => {
    const loose = join(work, "loose.png");
    const tight = join(work, "tight.png");
    const a = shrinkPng(photo, loose, { maxBytes: statSync(photo).size, tools });
    const b = shrinkPng(photo, tight, { maxBytes: 1000, minColours: 32, tools });
    expect(pngSize(readFileSync(loose))).toEqual({ width: 320, height: 200 });
    expect(pngSize(readFileSync(tight))).toEqual({ width: 320, height: 200 });
    expect(a.underTarget).toBe(true);
    expect(statSync(loose).size).toBeLessThan(statSync(photo).size);
    // The tight target cannot be met; the tool stops at the fewest colours it may use and says so.
    expect(b.underTarget).toBe(false);
    expect(b.colours).toBeLessThanOrEqual(a.colours);
    expect(b.colours).toBeGreaterThanOrEqual(32);
    expect(statSync(tight).size).toBeLessThanOrEqual(statSync(loose).size);
  });

  it.skipIf(!can)("keeps the picture as it is when asked for lossless", () => {
    const out = join(work, "lossless.png");
    const r = shrinkPng(photo, out, { maxBytes: 10, tools, lossless: true });
    expect(r.lossless).toBe(true);
    expect(pngSize(readFileSync(out))).toEqual({ width: 320, height: 200 });
  });

  it.skipIf(!can)("halves a picture, and finds where it is flat", () => {
    const small = join(work, "small.png");
    halve(photo, small, tools);
    expect(pngSize(readFileSync(small))).toEqual({ width: 160, height: 100 });
    expect(spread(photo, { x: 0, y: 0, width: 320, height: 200 }, tools)).toBeGreaterThan(0.02);
    const flat = join(work, "flat.png");
    spawnSync(tools.magick, ["-size", "100x100", "xc:white", flat]);
    expect(spread(flat, { x: 0, y: 0, width: 100, height: 100 }, tools)).toBeLessThan(0.001);
  });
});

describe("shrinking a screenshot: photographs and interface apart", () => {
  const tools = findTools();
  const can = !!tools.magick && !!tools.pngquant;
  let shot;
  // 400x300: a cyan interface with a small yellow patch (the moon mark), a noisy "photograph" in a rectangle and a small gradient (a colour wheel).
  const photo = { x: 100, y: 80, width: 280, height: 200 };
  const gradient = { x: 10, y: 200, width: 80, height: 40 };
  beforeAll(() => {
    if (!can) return;
    shot = join(work, "shot.png");
    const r = spawnSync(tools.magick, ["-size", "400x300", "xc:#0DB7E8", "-fill", "#FFC20E", "-draw", "circle 30,30 30,12", "(", "-size", "280x200", "-seed", "3", "plasma:fractal", ")", "-geometry", "+100+80", "-composite", "(", "-size", "80x40", "gradient:red-blue", ")", "-geometry", "+10+200", "-composite", shot]);
    if (r.status !== 0) throw new Error("ImageMagick could not make a test screenshot: " + r.stderr);
  });

  it.skipIf(!can)("lists the most frequent colours first", () => {
    const colours = topColours(shot, 2, tools);
    expect(colours[0]).toEqual([0x0d, 0xb7, 0xe8]);
    expect(colours).toHaveLength(2);
  });

  it("scales rectangles outwards, never inwards", () => {
    expect(scaleRects([{ x: 3, y: 5, width: 11, height: 7 }], 0.5)).toEqual([{ x: 1, y: 2, width: 6, height: 4 }]);
  });

  it.skipIf(!can)("keeps the interface's exact colours, however small the patch, and writes an 8-bit PNG of at most 256 colours", () => {
    const out = join(work, "shot-small.png");
    const r = shrinkShot(shot, out, { maxBytes: 20000, minColours: 16, tools, photoRects: [photo] });
    const bytes = readFileSync(out);
    expect(pngSize(bytes)).toEqual({ width: 400, height: 300 });
    // The PNG header's colour type: 3 is a palette.
    expect(bytes[25]).toBe(3);
    expect(r.colours).toBeLessThanOrEqual(256);
    expect(r.interfaceColours).toBeGreaterThan(0);
    const at = (x, y) => spawnSync(tools.magick, [out, "-crop", `1x1+${x}+${y}`, "+repage", "-depth", "8", "-format", "%[hex:p{0,0}]", "info:"], { encoding: "utf8" }).stdout.trim().toUpperCase();
    expect(at(30, 30)).toBe("FFC20E");
    expect(at(5, 150)).toBe("0DB7E8");
    // A photograph left alone would have kept all of its colours; the shrinking cut them.
    expect(r.photoColours).toBeLessThan(256 - r.interfaceColours + 1);
  });

  it("reads the colours of a piece of CSS, each once", () => {
    expect(cssColours("--a: #FFC20E; --b: #0db7e8;\n --c: #ffc20e; border: 2px solid #000; color: #12345678;")).toEqual([
      [0xff, 0xc2, 0x0e],
      [0x0d, 0xb7, 0xe8],
    ]);
  });

  it.skipIf(!can)("keeps a small flat colour when it is asked to, even where the most frequent colours do not reach it", () => {
    const yellow = [0xff, 0xc2, 0x0e];
    const at = (file, x, y) => spawnSync(tools.magick, [file, "-crop", `1x1+${x}+${y}`, "+repage", "-depth", "8", "-format", "%[hex:p{0,0}]", "info:"], { encoding: "utf8" }).stdout.trim().toUpperCase();
    // Two palette places for the interface (black, from the photograph's place, and cyan): the 5 px yellow patch is not one of the two most
    // frequent colours, as the moon mark is not once a picture has been halved.
    const small = join(work, "tiny-moon.png");
    const r = spawnSync(tools.magick, ["-size", "400x300", "xc:#0DB7E8", "-fill", "#FFC20E", "-draw", "rectangle 20,20 24,24", "(", "-size", "280x200", "-seed", "3", "plasma:fractal", ")", "-geometry", "+100+80", "-composite", small]);
    if (r.status !== 0) throw new Error("ImageMagick could not make a test screenshot: " + r.stderr);
    const without = join(work, "tiny-moon-without.png");
    const withKeep = join(work, "tiny-moon-kept.png");
    shrinkShot(small, without, { maxBytes: 1000000, minColours: 16, tools, photoRects: [photo], uiColours: 2 });
    const kept = shrinkShot(small, withKeep, { maxBytes: 1000000, minColours: 16, tools, photoRects: [photo], uiColours: 2, keepColours: [yellow, [1, 2, 3]] });
    expect(at(without, 22, 22)).not.toBe("FFC20E");
    expect(at(withKeep, 22, 22)).toBe("FFC20E");
    // A colour to keep that is not on the picture costs no palette place.
    expect(kept.interfaceColours).toBe(3);
  });

  it.skipIf(!can)("gives a gradient a palette of its own, so a small one does not turn into a few flat colours", () => {
    const colours = (file, r) => Number(spawnSync(tools.magick, [file, "-crop", `${r.width}x${r.height}+${r.x}+${r.y}`, "+repage", "-format", "%k", "info:"], { encoding: "utf8" }).stdout);
    const without = join(work, "shot-no-gradient.png");
    const withGradient = join(work, "shot-gradient.png");
    // A tiny interface palette, as the real one is for a gradient with more colours than it has room for.
    shrinkShot(shot, without, { maxBytes: 1000000, tools, photoRects: [photo], uiColours: 4 });
    const r = shrinkShot(shot, withGradient, { maxBytes: 1000000, tools, photoRects: [photo], gradientRects: [gradient], gradientColours: 24, uiColours: 4 });
    expect(r.gradientColours).toBe(24);
    expect(colours(without, gradient)).toBeLessThanOrEqual(4);
    expect(colours(withGradient, gradient)).toBeGreaterThan(colours(without, gradient));
    expect(colours(withGradient, gradient)).toBeGreaterThanOrEqual(12);
  });

  it.skipIf(!can)("falls back to one palette when it is not told where the photographs are", () => {
    const out = join(work, "shot-plain.png");
    const r = shrinkShot(shot, out, { maxBytes: 1000000, tools, photoRects: [] });
    expect(r.lossless).toBe(false);
    expect(pngSize(readFileSync(out))).toEqual({ width: 400, height: 300 });
  });
});

describe("the shared server of the real shell", () => {
  let server;
  let reports;
  let port;
  const fetchText = (path) =>
    new Promise((resolve, reject) => {
      get({ host: "127.0.0.1", port, path }, (res) => {
        let body = "";
        res.on("data", (d) => (body += d));
        res.on("end", () => resolve({ status: res.statusCode, body }));
      }).on("error", reject);
    });

  beforeAll(async () => {
    const dist = join(work, "dist");
    mkdirSync(dist, { recursive: true });
    writeFileSync(join(dist, "index.html"), "<html><head><title>app</title></head><body></body></html>");
    reports = [];
    server = serveFrontend({
      dist,
      port: 0,
      driver: "window.__driver = true;",
      config: { answer: 42 },
      onReport: async (kind, payload) => {
        await new Promise((r) => setTimeout(r, 30));
        reports.push({ kind, payload });
      },
    });
    // Port 0: any free port (the real shell needs 1420, a test must not).
    await listen(server, 0);
    port = server.address().port;
  });
  afterAll(() => server.close());

  it("adds the config, the kit and the driver to index.html", async () => {
    const page = await fetchText("/");
    expect(page.status).toBe(200);
    expect(page.body).toContain('window.__SMOKE__ = {"answer":42}');
    expect(page.body.indexOf("/__kit.js")).toBeLessThan(page.body.indexOf("/__driver.js"));
    expect((await fetchText("/__driver.js")).body).toBe("window.__driver = true;");
    const kit = await fetchText("/__kit.js");
    expect(kit.body).toContain("window.__YM_KIT__");
  });

  it("joins the pieces of a long report and answers only after it was handled", async () => {
    const text = JSON.stringify({ name: "photo-look", note: "é".repeat(30) });
    const half = Math.floor(text.length / 2);
    const piece = (i, data) => `/__report?kind=shot&id=abc&i=${i}&n=2&data=${encodeURIComponent(data)}`;
    const first = await fetchText(piece(0, text.slice(0, half)));
    expect(first.status).toBe(204);
    expect(reports).toEqual([]);
    const second = await fetchText(piece(1, text.slice(half)));
    expect(second.status).toBe(204);
    expect(reports).toEqual([{ kind: "shot", payload: { name: "photo-look", note: "é".repeat(30) } }]);
  });

  it("does not serve files outside dist", async () => {
    expect((await fetchText("/../package.json")).status).toBe(404);
    expect((await fetchText("/missing.js")).status).toBe(404);
  });
});

it("links a photo the way the runner does", () => {
  // The app is given a folder of links when a name is asked for: a link to an ARW file is a file to the core's directory listing.
  const target = join(work, "real.ARW");
  writeFileSync(target, "x");
  const folder = join(work, "Named");
  mkdirSync(folder, { recursive: true });
  symlinkSync(target, join(folder, "real.ARW"));
  expect(findPhotos([folder]).map((f) => f.split("/").pop())).toEqual(["real.ARW"]);
});
