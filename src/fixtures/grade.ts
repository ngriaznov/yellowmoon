/**
 * What the mock "renders". The design mockups draw each photo as flat shapes and fake a look,
 * an exposure change and a white balance change by mixing the four colours of a scene towards
 * other colours (`design/mockups/photos.html`, function `grade`). This file is that function,
 * made reusable: the mock API uses it to make thumbnails, and the browser build of the Photos
 * screen uses it to draw the photo cell (the app draws it with the native viewport instead).
 *
 * Nothing here is colour science. It exists so that screenshots of the app can be compared with
 * the mockups' screenshots, and so that a change in a control is visible in a browser.
 */
import { sceneBody, SVG_HEIGHT, SVG_WIDTH, SUN_FRACTION_PHOTO } from "./scenes";
import type { Scene } from "./scenes";

type Rgb = [number, number, number];

const hx = (hex: string): Rgb => {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const toHex = (c: readonly number[]): string =>
  "#" + c.map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0")).join("").toUpperCase();
const mix = (a: string, b: string, t: number): string => {
  const A = hx(a);
  const B = hx(b);
  return toHex(A.map((x, i) => x + ((B[i] as number) - x) * t));
};
const gray = (a: string): string => {
  const [r, g, b] = hx(a);
  const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return toHex([y, y, y]);
};

/** Parameters of one look's flat-shape grading in the mockup. */
export interface LookGrade {
  /** Highlight tint and how much of it goes into the sky. */
  hi: string;
  ha: number;
  /** Shadow tint and how much of it goes into the ground. */
  sh: string;
  sa: number;
  /** Desaturation, 0–1. */
  ds: number;
}

/** The looks of the mockup by slug (the last part of the look id, `lookSlug`). Unknown slugs are neutral. */
export const LOOK_GRADES: Readonly<Record<string, LookGrade>> = {
  neutral: { hi: "#FFFFFF", ha: 0, sh: "#000000", sa: 0, ds: 0 },
  signature: { hi: "#E9C49A", ha: 0.1, sh: "#2A3B44", sa: 0.18, ds: 0.08 },
  "film-warm": { hi: "#F0B37A", ha: 0.22, sh: "#4A3424", sa: 0.2, ds: 0.05 },
  "film-cool": { hi: "#BFD3E6", ha: 0.18, sh: "#1E3448", sa: 0.28, ds: 0.05 },
  "bleach-bypass": { hi: "#F2F2F0", ha: 0.12, sh: "#1A1C1E", sa: 0.25, ds: 0.6 },
  "teal-and-orange": { hi: "#F0A86A", ha: 0.25, sh: "#0F4A57", sa: 0.4, ds: 0 },
  "soft-print": { hi: "#F4E6D2", ha: 0.25, sh: "#5A5A60", sa: 0.25, ds: 0.15 },
  "night-blue": { hi: "#6C88B8", ha: 0.35, sh: "#0E1A33", sa: 0.45, ds: 0.2 },
  monochrome: { hi: "#FFFFFF", ha: 0, sh: "#000000", sa: 0.05, ds: 1 },
};

const KEYS = ["sky", "far", "near", "sun"] as const;

/** A scene seen through a look at strength `k` (0–1). */
export function gradeScene(scene: Scene, look: LookGrade, k = 1): Scene {
  const out: Scene = { ...scene };
  for (const key of KEYS) out[key] = mix(scene[key], gray(scene[key]), look.ds * k);
  out.sky = mix(out.sky, look.hi, look.ha * k);
  out.sun = mix(out.sun, look.hi, look.ha * k * 0.5);
  out.far = mix(mix(out.far, look.hi, look.ha * k * 0.6), look.sh, look.sa * k * 0.3);
  out.near = mix(out.near, look.sh, look.sa * k);
  return out;
}

/** The Before view: the scene flattened towards grey, like a camera-log signal. */
export function flatScene(scene: Scene): Scene {
  const flat = (c: string): string => mix(c, "#7A7F84", 0.45);
  return { ...scene, sky: flat(scene.sky), far: flat(scene.far), near: flat(scene.near), sun: flat(scene.sun) };
}

/** How far a photo's exposure and white balance are from what Automatic chose for it. */
export interface Adjustment {
  /** Exposure now and as Automatic has it, in EV. */
  ev: number;
  autoEv: number;
  /** White balance now and as Automatic has it. */
  kelvin: number;
  autoKelvin: number;
  tint: number;
  autoTint: number;
}

/**
 * Warms, cools, tints and brightens the four colours in proportion to the distance from the
 * automatic values: 4000 K is a full step of warmth, 50 tint a quarter of magenta or green,
 * one EV about a fifth of the way to white or black.
 */
export function adjustScene(scene: Scene, a: Adjustment): Scene {
  const dk = (a.kelvin - a.autoKelvin) / 4000;
  const warm = dk > 0 ? "#FFB46B" : "#6BA6FF";
  const warmAmount = Math.min(Math.abs(dk) * 0.35, 0.45);
  const dt = a.tint - a.autoTint;
  const tintColor = dt > 0 ? "#E070C8" : "#60D080";
  const tintAmount = Math.min((Math.abs(dt) / 50) * 0.25, 0.25);
  const de = a.ev - a.autoEv;
  const light = de > 0 ? "#FFFFFF" : "#000000";
  const lightAmount = Math.min(Math.abs(de) * 0.22, 0.75);
  const out: Scene = { ...scene };
  for (const key of KEYS) {
    let c = out[key];
    c = mix(c, warm, warmAmount);
    c = mix(c, tintColor, tintAmount);
    c = mix(c, light, lightAmount);
    out[key] = c;
  }
  return out;
}

/** What `developScene` needs to know. */
export interface DevelopLook {
  /** Slug of the look ("film-warm"). */
  slug: string;
  /** 0–100. */
  intensity: number;
}

/** A scene as the Photos screen shows it: adjusted, then through the look. */
export function developScene(scene: Scene, look: DevelopLook, adjustment: Adjustment | null): Scene {
  const adjusted = adjustment ? adjustScene(scene, adjustment) : scene;
  const grade = LOOK_GRADES[look.slug] ?? LOOK_GRADES.neutral;
  return gradeScene(adjusted, grade as LookGrade, Math.max(0, Math.min(100, look.intensity)) / 100);
}

/** A crop for `previewSvg`: a rectangle of the oriented frame in [0, 1]², and a straighten angle. */
export interface PreviewCrop {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** What `previewSvg` draws besides the scene. */
export interface PreviewOptions {
  /** Orientation in degrees clockwise: 0, 90, 180 or 270. */
  orientation?: number;
  /** Straighten angle in degrees, positive clockwise, about the centre of the frame. */
  angle_deg?: number;
  /** The visible part of the oriented frame. Default: all of it. */
  crop?: PreviewCrop | null;
  /** Sun size as a fraction of the frame width. */
  sunFraction?: number;
}

/**
 * The scene as an SVG document for the photo cell of a browser build: rotated by the
 * orientation, straightened, and cropped. The `<img>` letterboxes it on paper.
 */
export function previewSvg(scene: Scene, options: PreviewOptions = {}): string {
  const { orientation = 0, angle_deg = 0, crop = null, sunFraction = SUN_FRACTION_PHOTO } = options;
  const turned = orientation % 180 !== 0;
  const cw = turned ? SVG_HEIGHT : SVG_WIDTH;
  const ch = turned ? SVG_WIDTH : SVG_HEIGHT;
  const place =
    orientation === 90
      ? `translate(${SVG_HEIGHT} 0) rotate(90)`
      : orientation === 180
        ? `translate(${SVG_WIDTH} ${SVG_HEIGHT}) rotate(180)`
        : orientation === 270
          ? `translate(0 ${SVG_WIDTH}) rotate(270)`
          : "";
  const view = crop ?? { x: 0, y: 0, width: 1, height: 1 };
  const vx = view.x * cw;
  const vy = view.y * ch;
  const vw = Math.max(1, view.width * cw);
  const vh = Math.max(1, view.height * ch);
  const straighten = angle_deg ? `rotate(${angle_deg} ${cw / 2} ${ch / 2})` : "";
  const round = (n: number): string => String(Math.round(n * 100) / 100);
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${round(vw)}" height="${round(vh)}" viewBox="${round(vx)} ${round(vy)} ${round(vw)} ${round(vh)}">`,
    `<g transform="${straighten}"><g transform="${place}">`,
    ...sceneBody(scene, sunFraction),
    `</g></g>`,
    `</svg>`,
  ].join("");
}

/** `previewSvg` as a data URL, for an `<img src>`. */
export function previewDataUrl(scene: Scene, options: PreviewOptions = {}): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(previewSvg(scene, options))}`;
}
