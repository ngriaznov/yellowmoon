// @vitest-environment node
// The browser-side latency harness (`e2e/perf/*.perf.ts`, docs/11 Large folders, docs/13) is not part of `npm run e2e` (its `testMatch` is
// `*.spec.ts`, and it needs the production build), so nothing in the normal run notices when the app stops drawing what it selects: the
// filmstrip's counter cell became a caption in the dark glass interface, and the harness kept waiting for the old cell until the weekly
// job failed. This reads the harness as text and holds every class it names to the markup of the app.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const ROOT = new URL("../", import.meta.url).pathname;

/** Files under `dir` (recursively) whose names pass `keep`. */
function filesUnder(dir: string, keep: (name: string) => boolean): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return entry.name === "node_modules" ? [] : filesUnder(path, keep);
    return keep(entry.name) ? [path] : [];
  });
}

/** The markup and code of the app: not its tests, which may still say what must be gone. */
const appText = filesUnder(join(ROOT, "src"), (name) => /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name))
  .map((path) => readFileSync(path, "utf8"))
  .join("\n");

const harnesses = filesUnder(join(ROOT, "e2e", "perf"), (name) => name.endsWith(".perf.ts"));

/** Every `.ym-…` class a file selects. */
const classesOf = (text: string): string[] => [...new Set([...text.matchAll(/\.(ym-[\w-]+)/g)].map((m) => m[1] as string))];

/** True when `name` appears in the app as a whole class name (`ym-filmstrip__count` is not `ym-filmstrip__counter`). */
const drawn = (name: string): boolean => new RegExp(`(?<![\\w-])${name}(?![\\w-])`).test(appText);

describe("the perf harness names only classes the app draws", () => {
  it("finds the harness and the classes it selects", () => {
    expect(harnesses.length).toBeGreaterThan(0);
    expect(classesOf(harnesses.map((path) => readFileSync(path, "utf8")).join("\n"))).toContain("ym-filmstrip__caption");
  });

  it.each(harnesses.map((path) => [path.slice(ROOT.length), path]))("%s", (_name, path) => {
    const missing = classesOf(readFileSync(path as string, "utf8")).filter((name) => !drawn(name));
    expect(missing, "classes the app no longer draws; update the harness").toEqual([]);
  });
});
