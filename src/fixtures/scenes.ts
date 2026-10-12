/**
 * The twelve "Lisbon" photos of the design mockups (`design/mockups/photos.html`): flat shapes,
 * no real images. Each scene is a sky, a sun, a far hill (a half ellipse) and a near ground
 * band. This file is the single source of truth for them: the SVG files next to it are
 * generated from `sceneSvg` (a test keeps them in step), and the mock API rasterises the
 * same scenes into PNG thumbnails.
 *
 * Geometry, copied from the mockup's CSS (percentages of the 3:2 photo cell):
 *  - sky: the whole frame;
 *  - sun: a circle 9% wide (12% in the filmstrip), left `sx`%, top `sy`%;
 *  - far: a box from -10% to 110% across, bottom 18%, height 42%, rounded as a half ellipse;
 *  - near: a full-width band, bottom 0, height 26%.
 *
 * No imports, so Node can run this file directly to regenerate the SVGs.
 */

/** One flat-shape photo. Colours are `#RRGGBB`. */
export interface Scene {
  /** File name without extension: "DSC04201". */
  name: string;
  sky: string;
  far: string;
  near: string;
  sun: string;
  /** Sun position: percent of the frame width (left edge of the sun). */
  sx: number;
  /** Sun position: percent of the frame height (top edge of the sun). */
  sy: number;
}

/** The twelve photos, in file order. */
export const SCENES: readonly Scene[] = [
  { name: "DSC04201", sky: "#5B7088", far: "#34465A", near: "#1F2A36", sun: "#E8B47A", sx: 70, sy: 22 },
  { name: "DSC04202", sky: "#C99A63", far: "#7A5A3A", near: "#3E2F22", sun: "#F3D49A", sx: 30, sy: 28 },
  { name: "DSC04203", sky: "#9DB0A4", far: "#4C6450", near: "#2A3A2C", sun: "#E9E3C8", sx: 78, sy: 18 },
  { name: "DSC04204", sky: "#7FA6C7", far: "#E6ECF2", near: "#3F5468", sun: "#FFF6E0", sx: 66, sy: 12 },
  { name: "DSC04205", sky: "#8FB3C9", far: "#5E7F94", near: "#C8B79A", sun: "#F6EED8", sx: 60, sy: 14 },
  { name: "DSC04206", sky: "#241B33", far: "#3A2350", near: "#120E1C", sun: "#E86FA0", sx: 74, sy: 30 },
  { name: "DSC04207", sky: "#D7A877", far: "#A86F45", near: "#6E4630", sun: "#F5D9A8", sx: 45, sy: 20 },
  { name: "DSC04208", sky: "#6A5141", far: "#8C6A50", near: "#3A2C22", sun: "#F2C98B", sx: 50, sy: 30 },
  { name: "DSC04209", sky: "#8E959C", far: "#5F676E", near: "#3B4045", sun: "#C9CDD1", sx: 82, sy: 26 },
  { name: "DSC04210", sky: "#8C86A8", far: "#5B5677", near: "#2F2C40", sun: "#EAD2C0", sx: 36, sy: 24 },
  { name: "DSC04211", sky: "#1B2230", far: "#2B3446", near: "#12161F", sun: "#D9C9A0", sx: 20, sy: 16 },
  { name: "DSC04212", sky: "#B9C7D4", far: "#7E9A5E", near: "#4F6B35", sun: "#FBF2D5", sx: 24, sy: 18 },
];

/** Width of the SVG fixtures in user units; the height is two thirds of it. */
export const SVG_WIDTH = 1500;
/** Height of the SVG fixtures in user units (3:2). */
export const SVG_HEIGHT = 1000;

/** Sun diameter as a fraction of the frame width in the photo cell. */
export const SUN_FRACTION_PHOTO = 0.09;
/** Sun diameter as a fraction of the frame width in the filmstrip thumbnails. */
export const SUN_FRACTION_THUMB = 0.12;

const FAR_LEFT = -0.1;
const FAR_RIGHT = 1.1;
const FAR_BOTTOM = 0.18;
const FAR_HEIGHT = 0.42;
const NEAR_HEIGHT = 0.26;

/** Rounds to three decimals so that floating-point noise never reaches a file. */
const num = (x: number): string => String(Math.round(x * 1000) / 1000);

/** The shapes of the scene as SVG elements, one per line, in the frame of `SVG_WIDTH` × `SVG_HEIGHT`. */
export function sceneBody(scene: Scene, sunFraction: number = SUN_FRACTION_PHOTO): string[] {
  const w = SVG_WIDTH;
  const h = SVG_HEIGHT;
  const r = (sunFraction * w) / 2;
  const cx = (scene.sx / 100) * w + r;
  const cy = (scene.sy / 100) * h + r;
  const farRx = ((FAR_RIGHT - FAR_LEFT) * w) / 2;
  const farRy = FAR_HEIGHT * h;
  const farX0 = FAR_LEFT * w;
  const farX1 = FAR_RIGHT * w;
  const farBase = h - FAR_BOTTOM * h;
  const nearTop = h - NEAR_HEIGHT * h;
  return [
    `<rect width="${w}" height="${h}" fill="${scene.sky}"/>`,
    `<circle cx="${num(cx)}" cy="${num(cy)}" r="${num(r)}" fill="${scene.sun}"/>`,
    `<path d="M${num(farX0)} ${num(farBase)}A${num(farRx)} ${num(farRy)} 0 0 1 ${num(farX1)} ${num(farBase)}Z" fill="${scene.far}"/>`,
    `<rect y="${num(nearTop)}" width="${w}" height="${num(h - nearTop)}" fill="${scene.near}"/>`,
  ];
}

/** The scene as an SVG document, 3:2, using the geometry of the mockup's photo cell. */
export function sceneSvg(scene: Scene, sunFraction: number = SUN_FRACTION_PHOTO): string {
  const w = SVG_WIDTH;
  const h = SVG_HEIGHT;
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    ...sceneBody(scene, sunFraction),
    `</svg>`,
    ``,
  ].join("\n");
}

/** Which shape of the scene covers a point. */
export type SceneLayer = "sky" | "sun" | "far" | "near";

/**
 * The shape on top at a point of the frame, with `u` and `v` in [0, 1] across and down.
 * Used for pixel samples and for rasterising thumbnails; `aspect` is width / height.
 */
export function layerAt(scene: Scene, u: number, v: number, sunFraction: number = SUN_FRACTION_PHOTO, aspect = 1.5): SceneLayer {
  // Work in units of the frame height so that circles stay round.
  const x = u * aspect;
  const y = v;
  if (y >= 1 - NEAR_HEIGHT) return "near";
  const farCx = aspect / 2;
  const farRx = ((FAR_RIGHT - FAR_LEFT) * aspect) / 2;
  const farBase = 1 - FAR_BOTTOM;
  if (y <= farBase) {
    const nx = (x - farCx) / farRx;
    const ny = (y - farBase) / FAR_HEIGHT;
    if (nx * nx + ny * ny <= 1) return "far";
  }
  const r = (sunFraction * aspect) / 2;
  const sx = (scene.sx / 100) * aspect + r;
  const sy = scene.sy / 100 + r;
  if ((x - sx) ** 2 + (y - sy) ** 2 <= r * r) return "sun";
  return "sky";
}
