/**
 * Fixture data for the dev pages (`components.html`): the twelve "Lisbon" scenes from the mockups,
 * the look grading the mockup uses for its thumbnails, a histogram and curve points that match
 * the mockup's drawings. Nothing here ships in the app; it only feeds the component states.
 */
import type { CurvePoint } from "../api/types";
import type { FilmstripPhoto } from "../components";
import { SCENES, SUN_FRACTION_THUMB, sceneSvg } from "../fixtures/scenes";
import type { Scene } from "../fixtures/scenes";
import { fixtureUrl } from "../fixtures";
import { flatScene, gradeScene, logScene } from "../fixtures/grade";
import type { LookGrade } from "../fixtures/grade";

/** The scene of a photo name, "DSC04204". */
export function sceneNamed(name: string): Scene {
  return SCENES.find((s) => s.name === name) ?? (SCENES[3] as Scene);
}

/** A scene as a small SVG data URL (used for look thumbnails and filmstrip thumbnails). */
export function sceneDataUrl(scene: Scene, sunFraction = SUN_FRACTION_THUMB): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(sceneSvg(scene, sunFraction))}`;
}

// ---- Look grading (the mockup's flat-shape grade, shared with the mock API) -------------------

export type { LookGrade };
export { flatScene, gradeScene, logScene };

/** A look in the demo library. */
export interface DemoLook {
  id: string;
  name: string;
  group: "builtin" | "arri";
  grade: LookGrade;
}

export const DEMO_LOOKS: readonly DemoLook[] = [
  { id: "neutral", name: "Neutral", group: "builtin", grade: { hi: "#FFFFFF", ha: 0, sh: "#000000", sa: 0, ds: 0 } },
  { id: "signature", name: "Signature", group: "builtin", grade: { hi: "#E9C49A", ha: 0.1, sh: "#2A3B44", sa: 0.18, ds: 0.08 } },
  { id: "warm", name: "Film warm", group: "arri", grade: { hi: "#F0B37A", ha: 0.22, sh: "#4A3424", sa: 0.2, ds: 0.05 } },
  { id: "cool", name: "Film cool", group: "arri", grade: { hi: "#BFD3E6", ha: 0.18, sh: "#1E3448", sa: 0.28, ds: 0.05 } },
  { id: "bleach", name: "Bleach bypass", group: "arri", grade: { hi: "#F2F2F0", ha: 0.12, sh: "#1A1C1E", sa: 0.25, ds: 0.6 } },
  { id: "teal", name: "Teal and orange", group: "arri", grade: { hi: "#F0A86A", ha: 0.25, sh: "#0F4A57", sa: 0.4, ds: 0 } },
  { id: "soft", name: "Soft print", group: "arri", grade: { hi: "#F4E6D2", ha: 0.25, sh: "#5A5A60", sa: 0.25, ds: 0.15 } },
  { id: "night", name: "Night blue", group: "arri", grade: { hi: "#6C88B8", ha: 0.35, sh: "#0E1A33", sa: 0.45, ds: 0.2 } },
  { id: "mono", name: "Monochrome", group: "arri", grade: { hi: "#FFFFFF", ha: 0, sh: "#000000", sa: 0.05, ds: 1 } },
];

/** Thumbnail URL of the current photo through a look. */
export const lookThumbnail = (photo: string, look: DemoLook): string => sceneDataUrl(gradeScene(sceneNamed(photo), look.grade), 0.12);

// ---- Photos ---------------------------------------------------------------------------------

/** The mockup's picked photos: 3rd, 4th and 5th. */
export const PICKED_INDEXES: readonly number[] = [2, 3, 4];

/** Filmstrip photos of the Lisbon folder. */
export function lisbonPhotos(options: { picked?: readonly number[]; included?: readonly number[]; thumbs?: boolean } = {}): FilmstripPhoto[] {
  const picked = options.picked ?? PICKED_INDEXES;
  return SCENES.map((scene, index) => ({
    id: index + 1,
    name: scene.name,
    src: options.thumbs === false ? undefined : sceneDataUrl(scene),
    picked: picked.includes(index),
    included: options.included ? options.included.includes(index) : undefined,
  }));
}

/** Full-size photo URL of a scene. */
export const photoUrl = (name: string): string => fixtureUrl(name) ?? sceneDataUrl(sceneNamed(name), 0.09);

// ---- Histogram and curves --------------------------------------------------------------------

const bump = (x: number, centre: number, width: number): number => Math.exp(-((x - centre) ** 2) / (2 * width ** 2));

/** A plausible 64-bin RGB histogram, shaped like the mockup's drawing. */
export function demoHistogram(bins = 64): { red: number[]; green: number[]; blue: number[] } {
  const make = (shift: number): number[] =>
    Array.from({ length: bins }, (_, i) => {
      const x = i / (bins - 1);
      return 0.15 + 0.85 * (bump(x, 0.45 + shift, 0.18) + 0.45 * bump(x, 0.72 + shift, 0.14)) * (1 - 0.35 * x);
    });
  return { red: make(-0.02), green: make(0), blue: make(0.025) };
}

/** A single-series histogram for the Curve plot background. */
export const demoLumaHistogram = (): number[] => demoHistogram(96).green;

/** The mockup's luma curve: three points and the two ends (units of the plot). */
export const CURVE_DEMO_POINTS: Readonly<Record<"luma" | "red" | "green" | "blue", CurvePoint[]>> = {
  luma: [[0, 0], [0.25, 0.191], [0.5, 141 / 255], [0.75, 0.845], [1, 1]],
  red: [[0, 0], [0.25, 0.2], [0.5, 0.57], [0.75, 0.86], [1, 1]],
  green: [[0, 0], [0.5, 0.5], [1, 1]],
  blue: [[0, 0], [0.25, 0.1], [0.5, 0.45], [0.75, 0.82], [1, 1]],
};
