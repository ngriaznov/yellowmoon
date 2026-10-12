/**
 * URLs of the SVG fixtures (`DSC04201.svg` …), one per scene in `scenes.ts`. Vite turns each
 * import into a hashed asset URL, or a data URI for small files, which the app's CSP allows.
 */
import { SCENES } from "./scenes";

const urls = import.meta.glob<string>("./*.svg", { eager: true, query: "?url", import: "default" });

/** URL of the full-size SVG for a scene name ("DSC04204"), or undefined for an unknown name. */
export function fixtureUrl(name: string): string | undefined {
  return urls[`./${name}.svg`];
}

/** Names of every scene that has an SVG, in file order. */
export const FIXTURE_NAMES: readonly string[] = SCENES.map((s) => s.name);

export { SCENES, type Scene } from "./scenes";
