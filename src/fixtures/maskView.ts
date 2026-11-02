/**
 * The mask view on the mock's flat scene (`docs/23` §5.8): the SVG that stands in for the core's frame when a corrector's mask is shown.
 *
 * The core draws the real thing (a tint after output sharpening, or the grey code values `m`); the mock draws shapes: a Radial is an
 * ellipse that fades out over its feather, a Linear is a gradient across the photo, a Brush is its strokes (a round-capped line of the stroke's
 * size, softened by its hardness and the row's Feather; an erase stroke is black over the paint before it, and a row's Invert swaps the two),
 * and every other kind selects nothing. That is enough
 * for the interface tests and the reference screenshots, not for pictures. Rows add (a row painted over the others in white) or
 * subtract (in black), the row's Invert and the node's Invert swap the two, and a node with no row shows 1: the whole photo.
 */
import type { MaskItem, MaskSettings, MaskStyle } from "../api/types";

/** The mask accent (`--mask` in `design/tokens.css`). */
export const MASK_ACCENT = "#ff6fd2";
/** How far the overlay goes towards the accent where the mask is 1 (the core's 0.55). */
export const OVERLAY_WEIGHT = 0.55;

/**
 * The photo frame as the shapes live in it (`docs/23` §3.6), when it is not the picture itself: the photo's size in its own pixels (the
 * long edge is what sizes are fractions of) and the map `matrix(a b c d e f)` from those pixels to the frame's, which carries the turn and
 * the crop.
 */
export interface MaskPhoto {
  width: number;
  height: number;
  matrix: [number, number, number, number, number, number];
}

/** The picture's rectangle in the frame, in the frame's own pixels. */
export interface MaskRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

const round = (n: number): string => String(Math.round(n * 100) / 100);

/**
 * A Brush row on the flat scene: the strokes drawn in order into a plane of their own (black where nothing is painted, white where it is, an
 * erase stroke black again), which then paints the row's ink where it is white, at the row's opacity. A stroke is a polyline with round caps
 * as wide as the brush; its softness is a blur of a third of the ramp the core gives it (`docs/algorithms/06` §5.5: the ramp is `(1 − hardness)`
 * of the radius), and the row's Feather a blur of `feather · 0.02` of the long edge, the core's σ. Grow and Refine edges are not drawn. A row's
 * Invert draws the plane the other way round. The filters have their region given in the scene's units: a horizontal line has no height, and a
 * filter region of the default kind would be empty.
 */
function brushShape(row: Extract<MaskItem, { kind: "brush" }>, index: number, ink: string, box: MaskRect): string {
  const long = Math.max(box.width, box.height);
  const opacity = Math.max(0, Math.min(1, row.opacity));
  const id = `mvb${index}`;
  const region = `filterUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}"`;
  const paint = row.invert ? "#000" : "#fff";
  const clear = row.invert ? "#fff" : "#000";
  const blurs: string[] = [];
  const lines = row.strokes
    .map((stroke, k) => {
      const places = stroke.points.map(([x, y]) => `${round(box.x + x * box.width)},${round(box.y + y * box.height)}`);
      const first = places[0];
      if (first === undefined) return "";
      // One point is a dot: the same place twice, which a round cap draws.
      const points = places.length === 1 ? `${first} ${first}` : places.join(" ");
      const width = Math.max(1, stroke.size * long);
      const sigma = (1 - stroke.hardness) * (width / 2) * 0.33;
      let filter = "";
      if (sigma >= 0.2) {
        blurs.push(`<filter id="${id}s${k}" ${region}><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`);
        filter = ` filter="url(#${id}s${k})"`;
      }
      return `<polyline points="${points}" fill="none" stroke="${stroke.erase ? clear : paint}" stroke-width="${round(width)}" stroke-linecap="round" stroke-linejoin="round"${filter}/>`;
    })
    .join("");
  const feather = row.feather * 0.02 * long;
  let plane = `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${clear}"/>${lines}`;
  if (feather >= 0.2) {
    blurs.push(`<filter id="${id}f" ${region}><feGaussianBlur stdDeviation="${round(feather)}"/></filter>`);
    plane = `<g filter="url(#${id}f)">${plane}</g>`;
  }
  const mask = `<mask id="${id}" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${plane}</mask>`;
  return `<defs>${blurs.join("")}${mask}</defs><rect data-brush-row="" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}" mask="url(#${id})"/>`;
}

/** One row's gradient shape painted with `ink` (a luminance, "#fff" or "#000") at `m` and nothing outside; empty for a kind that selects nothing. */
function rowShape(row: MaskItem, index: number, ink: string, box: MaskRect): string {
  const long = Math.max(box.width, box.height);
  const id = `mv${index}`;
  const opacity = Math.max(0, Math.min(1, row.opacity));
  if (row.kind === "brush") return brushShape(row, index, ink, box);
  if (row.kind === "radial") {
    const [cx, cy] = row.center;
    const [rx, ry] = row.radii;
    const inner = Math.max(0, Math.min(1, 1 - row.feather));
    const hard = row.feather <= 0.001 ? 0.995 : inner;
    const stops = row.invert
      ? `<stop offset="${round(hard)}" stop-color="${ink}" stop-opacity="0"/><stop offset="1" stop-color="${ink}" stop-opacity="${opacity}"/>`
      : `<stop offset="${round(hard)}" stop-color="${ink}" stop-opacity="${opacity}"/><stop offset="1" stop-color="${ink}" stop-opacity="0"/>`;
    const at = `translate(${round(box.x + cx * box.width)} ${round(box.y + cy * box.height)}) rotate(${round(row.angle_deg)}) scale(${round(rx * long)} ${round(ry * long)})`;
    const body = `<circle cx="0" cy="0" r="1" fill="url(#${id})" transform="${at}"/>`;
    const outside = row.invert ? `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}"/>` : "";
    return `<defs><radialGradient id="${id}" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse">${stops}</radialGradient></defs>${outside}${body}`;
  }
  if (row.kind === "linear") {
    const [cx, cy] = row.center;
    const half = Math.max(0.002, row.feather) * long * 0.5;
    const x = box.x + cx * box.width;
    const y = box.y + cy * box.height;
    const a = (row.angle_deg * Math.PI) / 180;
    // The normal of the core's Linear (`docs/algorithms/06` §4.1): for an angle of 0 it points down, so the line is horizontal.
    const [nx, ny] = [-Math.sin(a), Math.cos(a)];
    const [x1, y1, x2, y2] = row.invert
      ? [x + nx * half, y + ny * half, x - nx * half, y - ny * half]
      : [x - nx * half, y - ny * half, x + nx * half, y + ny * half];
    const stops = `<stop offset="0" stop-color="${ink}" stop-opacity="${opacity}"/><stop offset="1" stop-color="${ink}" stop-opacity="0"/>`;
    return `<defs><linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="${round(x1)}" y1="${round(y1)}" x2="${round(x2)}" y2="${round(y2)}">${stops}</linearGradient></defs><rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="url(#${id})"/>`;
  }
  return "";
}

/**
 * The luminance mask of a node's rows over `box`: white where `m` is 1, black where it is 0. Rows that are off count as absent; a node
 * with no row at all is 1 everywhere, and its Invert makes it 0.
 */
function maskContent(masks: MaskSettings, box: MaskRect): string {
  const rows = masks.items.filter((row) => row.enabled);
  const flip = masks.invert;
  const paper = (ink: string): string => `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}"/>`;
  if (rows.length === 0) return paper(flip ? "#000" : "#fff");
  // A lone Subtract starts from the whole photo; every other list starts from nothing.
  const first = rows[0];
  const base = first?.mode === "subtract" ? "#fff" : "#000";
  const shapes = rows
    .map((row, i) => rowShape(row, i, row.mode === "subtract" ? "#000" : "#fff", box))
    .join("");
  const content = paper(base) + shapes;
  // The node's Invert is the same picture with the two inks exchanged.
  return flip ? `<g filter="url(#mvinvert)">${content}</g>` : content;
}

/**
 * The SVG that draws a mask view over the picture at `box` (in the frame's pixels): `Overlay` tints the accent where the mask is 1,
 * `Mask only` paints the mask as grey over black. An empty string for no view.
 */
export function maskViewSvg(style: MaskStyle, masks: MaskSettings, box: MaskRect, photo?: MaskPhoto): string {
  // The shapes are in the photo frame: drawn on a rectangle of the photo's own size and carried to the picture by its map (the turn and the crop).
  const content = photo
    ? `<g transform="matrix(${photo.matrix.map(round).join(" ")})">${maskContent(masks, { x: 0, y: 0, width: photo.width, height: photo.height })}</g>`
    : maskContent(masks, box);
  const mask = `<mask id="mvmask" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${content}</mask>`;
  const invert = `<filter id="mvinvert"><feColorMatrix type="matrix" values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter>`;
  const rect = (fill: string, extra: string): string =>
    `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${fill}" ${extra}/>`;
  if (style === "overlay") {
    return `<g data-mask-view="overlay"><defs>${invert}${mask}</defs>${rect(MASK_ACCENT, `fill-opacity="${OVERLAY_WEIGHT}" mask="url(#mvmask)"`)}</g>`;
  }
  return `<g data-mask-view="mask_only"><defs>${invert}${mask}</defs>${rect("#000", "")}${rect("#fff", 'mask="url(#mvmask)"')}</g>`;
}

/**
 * The map `matrix(a b c d e f)` from the photo's own pixels to a picture that is the photo turned by `orientation` (degrees clockwise, 0, 90, 180 or 270) on a
 * canvas of `turned` size: what the mock's fixture picture does with the scene (`previewSvg`'s `place`), for the shapes of the photo frame to follow it.
 */
export function turnMatrix(orientation: number, width: number, height: number): MaskPhoto["matrix"] {
  switch (((orientation % 360) + 360) % 360) {
    case 90:
      return [0, 1, -1, 0, height, 0];
    case 180:
      return [-1, 0, 0, -1, width, height];
    case 270:
      return [0, -1, 1, 0, 0, width];
    default:
      return [1, 0, 0, 1, 0, 0];
  }
}
