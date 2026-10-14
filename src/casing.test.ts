import { describe, expect, it } from "vitest";

// macOS and Windows file systems ignore case, so `import "./Wheel"` can resolve to `wheel.ts` there while Linux finds
// `Wheel.tsx`. The build broke on the macOS runner for exactly that pair; this keeps the frontend free of such pairs.
// The glob only lists paths (nothing is imported).
const files = Object.keys(import.meta.glob(["/src/**/*", "/e2e/**/*"]));

function collisions(paths: string[]): string[] {
  const stems = new Map<string, string>();
  const found: string[] = [];
  for (const path of paths) {
    const slash = path.lastIndexOf("/");
    const dir = path.slice(0, slash);
    const stem = path.slice(slash + 1).split(".")[0] ?? "";
    const key = `${dir}/${stem.toLowerCase()}`;
    const seen = stems.get(key);
    if (seen !== undefined && seen !== stem) found.push(`${dir}: ${seen} and ${stem}`);
    else stems.set(key, stem);
  }
  return found;
}

describe("file names", () => {
  it("lists the frontend's files", () => {
    expect(files).toContain("/src/components/Wheel.tsx");
  });

  it("no two modules in one folder differ only in case", () => {
    expect(collisions(files)).toEqual([]);
  });
});
