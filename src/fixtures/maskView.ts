/**
 * The mask view on the mock's flat scene (`docs/23` §5.8): the SVG that stands in for the core's frame when a corrector's mask is shown.
 *
 * The core draws the real thing (a tint after output sharpening, or the grey code values `m`); the mock draws shapes: a Radial is an
 * ellipse that fades out over its feather, a Linear is a gradient across the photo, a Brush is its strokes (a round-capped line of the stroke's
 * size, softened by its hardness and the row's Feather; an erase stroke is black over the paint before it, and a row's Invert swaps the two),
 * a Background is the flat scene's blur map cut by the row's Level and Softness (the sky is out of focus, the far hill is sharp, the near ground is
 * half way), an Object is the far hill (the scene's "mountain") with a round hole at every negative point and cut to the row's box, a Subject is the far hill
 * (Vision's "main thing"), a People is a figure standing on the near ground in the scenes that have one (the others find nothing), a Depth is the gradient of
 * its far limit over the scene's three depth positions (white at the near ground, none from the limit on), and every other kind selects nothing. That is enough
 * for the interface tests and the reference screenshots, not for pictures. Rows add (a row painted over the others in white) or
 * subtract (in black), the row's Invert and the node's Invert swap the two, and a node with no row shows 1: the whole photo.
 */
import type { DevelopSettings, MaskItem, MaskSettings, MaskStyle } from "../api/types";

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

/** The narrowest ramp of a Background's threshold, in probability (`docs/algorithms/06` §5.4). */
export const BLUR_RAMP_MIN = 0.05;

/** `S(t) = t²(3 − 2t)` for `t` held to [0, 1]. */
const smooth = (t: number): number => {
  const x = Number.isNaN(t) ? 0 : Math.max(0, Math.min(1, t));
  return x * x * (3 - 2 * x);
};

/**
 * A Background row's `S₀` for a cell whose probability of being out of focus is `p` (`docs/algorithms/06` §5.4): `S((P − (1 − λ) + w/2) / w)` with
 * `w = max(σ, 0.05)`, `λ` the row's `level` and `σ` its `softness`. The core does this on the cells of the photo's blur map; the mock on the three
 * areas of its flat scene.
 */
export function backgroundValue(p: number, level: number, softness: number): number {
  const w = Math.max(softness, BLUR_RAMP_MIN);
  return smooth((p - (1 - level) + 0.5 * w) / w);
}

/** The mock's blur map: `P` of the flat scene's three areas (`docs/23` §4.6 — the core's is the photo's own). The sky is out of focus, the far hill is sharp. */
export const MOCK_BLUR = { sky: 0.95, hill: 0.05, near: 0.55 } as const;

/** Whether the photo's graph has an Object row with the eye on: the mock then "makes ready" the photo's features (`object_state`). */
export const hasObjectRow = (settings: DevelopSettings): boolean =>
  settings.nodes.items.some((node) => node.kind === "corrector" && node.masks.items.some((row) => row.kind === "object" && row.enabled));

/**
 * The Subject and People rows with the eye on in the photo's graph, as the kinds the mock's Vision "works for" (`vision_state`, `viewport_busy.finding`), Subject
 * first and each once; none for a graph with no such row.
 */
export const visionRowKinds = (settings: DevelopSettings): Array<"subject" | "people"> => {
  const kinds = new Set<string>();
  for (const node of settings.nodes.items) {
    if (node.kind !== "corrector") continue;
    for (const row of node.masks.items) if (row.enabled && (row.kind === "subject" || row.kind === "people")) kinds.add(row.kind);
  }
  return (["subject", "people"] as const).filter((k) => kinds.has(k));
};

/** What the mock's Vision finds in a photo of the flat scenes (`docs/23` §4.5): a subject in every one (the far hill), people only in some. */
export interface VisionFound {
  subject: boolean;
  people: boolean;
}

/** The flat scenes that have a person standing on the near ground; Vision finds nothing for People in the others ("Nothing found"). */
export const SCENES_WITH_PEOPLE: ReadonlySet<string> = new Set(["DSC04202", "DSC04204", "DSC04205", "DSC04207", "DSC04208", "DSC04212"]);

/** What the mock's Vision finds in the scene with this name. */
export const visionFoundIn = (sceneName: string): VisionFound => ({ subject: true, people: SCENES_WITH_PEOPLE.has(sceneName) });

/** The person of the flat scenes that have one, in fractions of the picture: a head, and a body that stands on the near ground (`docs/23` §4.5). */
export const PERSON = { x: 0.3, headY: 0.58, headRadius: 0.035, bodyY: 0.76, bodyRadiusX: 0.055, bodyRadiusY: 0.16 } as const;

/** Whether the photo's graph has a Background row with the eye on: the mock then "makes" the photo's blur map on its first frame. */
export const hasBackgroundRow = (settings: DevelopSettings): boolean =>
  settings.nodes.items.some((node) => node.kind === "corrector" && node.masks.items.some((row) => row.kind === "background" && row.enabled));

/**
 * A Background row on the flat scene: the scene's three areas (the sky over the frame, the far hill — the half ellipse of `scenes.ts` — and the near
 * band) painted with their `S₀` as grey into a plane of their own, which paints the row's ink where it is lit, at the row's opacity, like a Brush's.
 * The map is a little soft (a blur of 0.6 % of the long edge, the map's cells being coarse) and the row's Feather a blur of `feather · 0.02` of it, the
 * core's σ. Grow and Refine edges are not drawn. A row's Invert draws the plane the other way round.
 */
function backgroundShape(row: Extract<MaskItem, { kind: "background" }>, index: number, ink: string, box: MaskRect): string {
  const long = Math.max(box.width, box.height);
  const opacity = Math.max(0, Math.min(1, row.opacity));
  const id = `mvg${index}`;
  const region = `filterUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}"`;
  const grey = (p: number): string => {
    const v = Math.round(255 * backgroundValue(p, row.level, row.softness));
    return `rgb(${v},${v},${v})`;
  };
  const at = (x: number, y: number): [string, string] => [round(box.x + x * box.width), round(box.y + y * box.height)];
  const [x0, y0] = at(0, 0);
  const [nx, ny] = at(0, 0.74);
  const [cx, cy] = at(0.5, 0.82);
  const rect = (x: string, y: string, h: number, fill: string): string => `<rect x="${x}" y="${y}" width="${round(box.width)}" height="${round(h)}" fill="${fill}"/>`;
  const map =
    rect(x0, y0, box.height, grey(MOCK_BLUR.sky)) +
    `<ellipse cx="${cx}" cy="${cy}" rx="${round(0.6 * box.width)}" ry="${round(0.42 * box.height)}" fill="${grey(MOCK_BLUR.hill)}"/>` +
    rect(nx, ny, 0.26 * box.height, grey(MOCK_BLUR.near));
  const blurs: string[] = [];
  const sigma = 0.006 * long + row.feather * 0.02 * long;
  let plane = map;
  if (sigma >= 0.2) {
    blurs.push(`<filter id="${id}f" ${region}><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`);
    plane = `<g filter="url(#${id}f)">${map}</g>`;
  }
  if (row.invert) {
    blurs.push(`<filter id="${id}i" ${region}><feColorMatrix type="matrix" values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter>`);
    plane = `<g filter="url(#${id}i)">${plane}</g>`;
  }
  const mask = `<mask id="${id}" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${plane}</mask>`;
  return `<defs>${blurs.join("")}${mask}</defs><rect data-background-row="" data-level="${round(row.level)}" data-softness="${round(row.softness)}" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}" mask="url(#${id})"/>`;
}

/** The far hill of the flat scene as the mock draws its masks: the half ellipse of `scenes.ts`, in fractions of the picture's box. */
export const HILL = { cx: 0.5, cy: 0.82, rx: 0.6, ry: 0.42 } as const;

/** The radius of the hole a negative point takes out of an Object row on the flat scene, as a fraction of the long edge. */
export const OBJECT_HOLE = 0.06;

/**
 * An Object row on the flat scene: the far hill (the scene's "mountain", the one thing the mock's `mask_find` finds) painted white into a plane of its own
 * (black elsewhere), a black hole at every negative point, the whole cut to the row's box when it has one. A row with no point and no box selects nothing.
 * The plane paints the row's ink where it is white, at the row's opacity, softened by the row's Feather like a Brush's. Grow and Refine edges are not drawn.
 * A row's Invert draws the plane the other way round.
 */
function objectShape(row: Extract<MaskItem, { kind: "object" }>, index: number, ink: string, box: MaskRect): string {
  if (row.points.length === 0 && !row.box) return "";
  const long = Math.max(box.width, box.height);
  const opacity = Math.max(0, Math.min(1, row.opacity));
  const id = `mvo${index}`;
  const region = `filterUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}"`;
  const at = (x: number, y: number): [string, string] => [round(box.x + x * box.width), round(box.y + y * box.height)];
  const [hx, hy] = at(HILL.cx, HILL.cy);
  // The near band (the bottom 26 % of the frame) is in front of the hill: black over it, as the scene draws it.
  const [nx, ny] = at(0, 0.74);
  const hill =
    `<ellipse cx="${hx}" cy="${hy}" rx="${round(HILL.rx * box.width)}" ry="${round(HILL.ry * box.height)}" fill="#fff"/>` +
    `<rect x="${nx}" y="${ny}" width="${round(box.width)}" height="${round(0.26 * box.height)}" fill="#000"/>`;
  const holes = row.points
    .filter(([, , positive]) => !positive)
    .map(([x, y]) => {
      const [cx, cy] = at(x, y);
      return `<circle cx="${cx}" cy="${cy}" r="${round(OBJECT_HOLE * long)}" fill="#000"/>`;
    })
    .join("");
  let plane = `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="#000"/>${hill}${holes}`;
  const blurs: string[] = [];
  const sigma = row.feather * 0.02 * long;
  if (sigma >= 0.2) {
    blurs.push(`<filter id="${id}f" ${region}><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`);
    plane = `<g filter="url(#${id}f)">${plane}</g>`;
  }
  if (row.box) {
    const [x0, y0, x1, y1] = row.box;
    const [bx, by] = at(x0, y0);
    const cut = `<rect x="${bx}" y="${by}" width="${round((x1 - x0) * box.width)}" height="${round((y1 - y0) * box.height)}" fill="#fff"/>`;
    plane = `<defs><mask id="${id}b" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${cut}</mask></defs><g mask="url(#${id}b)">${plane}</g>`;
  }
  if (row.invert) {
    blurs.push(`<filter id="${id}i" ${region}><feColorMatrix type="matrix" values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter>`);
    plane = `<g filter="url(#${id}i)">${plane}</g>`;
  }
  const mask = `<mask id="${id}" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${plane}</mask>`;
  return `<defs>${blurs.join("")}${mask}</defs><rect data-object-row="" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}" mask="url(#${id})"/>`;
}

/**
 * A Subject or People row on the flat scene: Vision's union of instances. Subject is the far hill (the scene's main thing, as the mock's Object finds it,
 * the near ground in front of it); People is the figure of the scenes that have one. A request that found nothing, or that this computer cannot make
 * (`found` false), selects nothing and paints no ink. The plane paints the row's ink where it is white, at the row's opacity, softened by the row's
 * Feather like a Brush's. Grow and Refine edges are not drawn. A row's Invert draws the plane the other way round.
 */
function visionShape(row: Extract<MaskItem, { kind: "subject" | "people" }>, index: number, ink: string, box: MaskRect, found: VisionFound): string {
  if (!found[row.kind]) return "";
  const long = Math.max(box.width, box.height);
  const opacity = Math.max(0, Math.min(1, row.opacity));
  const id = `mvv${index}`;
  const region = `filterUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}"`;
  const at = (x: number, y: number): [string, string] => [round(box.x + x * box.width), round(box.y + y * box.height)];
  let instances: string;
  if (row.kind === "subject") {
    const [hx, hy] = at(HILL.cx, HILL.cy);
    const [nx, ny] = at(0, 0.74);
    instances =
      `<ellipse cx="${hx}" cy="${hy}" rx="${round(HILL.rx * box.width)}" ry="${round(HILL.ry * box.height)}" fill="#fff"/>` +
      `<rect x="${nx}" y="${ny}" width="${round(box.width)}" height="${round(0.26 * box.height)}" fill="#000"/>`;
  } else {
    const [px, headY] = at(PERSON.x, PERSON.headY);
    const [, bodyY] = at(PERSON.x, PERSON.bodyY);
    instances =
      `<circle cx="${px}" cy="${headY}" r="${round(PERSON.headRadius * long)}" fill="#fff"/>` +
      `<ellipse cx="${px}" cy="${bodyY}" rx="${round(PERSON.bodyRadiusX * long)}" ry="${round(PERSON.bodyRadiusY * box.height)}" fill="#fff"/>`;
  }
  let plane = `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="#000"/>${instances}`;
  const blurs: string[] = [];
  const sigma = row.feather * 0.02 * long;
  if (sigma >= 0.2) {
    blurs.push(`<filter id="${id}f" ${region}><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`);
    plane = `<g filter="url(#${id}f)">${plane}</g>`;
  }
  if (row.invert) {
    blurs.push(`<filter id="${id}i" ${region}><feColorMatrix type="matrix" values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter>`);
    plane = `<g filter="url(#${id}i)">${plane}</g>`;
  }
  const mask = `<mask id="${id}" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${plane}</mask>`;
  return `<defs>${blurs.join("")}${mask}</defs><rect data-vision-row="${row.kind}" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}" mask="url(#${id})"/>`;
}

/**
 * The depth positions the mock's depth map gives the flat scene's three areas (`docs/algorithms/06` §5.8: 0 is the nearest part of the photo): the near ground, the far
 * hill and the sky. The core's map is the photo's own; this is enough for the interface tests and the reference screenshots.
 */
export const MOCK_DEPTH = { near: 0.15, far: 0.7, sky: 0.95 } as const;

/** The least far limit of a Depth row (`ym_color::mask_depth::DEPTH_FAR_MIN`). */
export const DEPTH_FAR_MIN = 0.02;
/** How far behind the picked depth the fade ends (`DEPTH_PICK_MARGIN` of the core): Pick depth and Focus plane set `far` to the depth plus this, at most 1. */
export const DEPTH_PICK_MARGIN = 0.1;

/** The far limit Pick depth and Focus plane set for a depth position `z` (`docs/algorithms/06` §5.8, step 4): `min(1, z + 0.1)`, not rounded. */
export const pickDepthFar = (z: number): number => Math.min(1, z + DEPTH_PICK_MARGIN);

/** The Depth row's mask at depth position `z` for the far limit `far` (§5.8, step 2): `clamp(1 − z / far, 0, 1)`, 1 at the nearest and none from `far` on. */
export function depthValue(z: number, far: number): number {
  const f = Number.isNaN(far) ? 1 : Math.min(1, Math.max(DEPTH_FAR_MIN, far));
  return Math.min(1, Math.max(0, 1 - z / f));
}

/**
 * A Depth row on the flat scene: the three areas (the sky over the frame, the far hill and the near band) painted with the row's gradient at their depth
 * positions (`MOCK_DEPTH`) as grey into a plane of their own, which paints the row's ink where it is lit, at the row's opacity, like a Background's. The map is a
 * little soft (a blur of 0.6 % of the long edge) and the row's Feather a blur of `feather · 0.02` of it. Grow and Refine edges are not drawn. A row's Invert draws
 * the plane the other way round. Until the map is made (`ready` false) the row selects nothing.
 */
function depthShape(row: Extract<MaskItem, { kind: "depth" }>, index: number, ink: string, box: MaskRect, ready: boolean): string {
  if (!ready) return "";
  const long = Math.max(box.width, box.height);
  const opacity = Math.max(0, Math.min(1, row.opacity));
  const id = `mvd${index}`;
  const region = `filterUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}"`;
  const grey = (z: number): string => {
    const v = Math.round(255 * depthValue(z, row.far));
    return `rgb(${v},${v},${v})`;
  };
  const at = (x: number, y: number): [string, string] => [round(box.x + x * box.width), round(box.y + y * box.height)];
  const [x0, y0] = at(0, 0);
  const [nx, ny] = at(0, 0.74);
  const [cx, cy] = at(HILL.cx, HILL.cy);
  const rect = (x: string, y: string, h: number, fill: string): string => `<rect x="${x}" y="${y}" width="${round(box.width)}" height="${round(h)}" fill="${fill}"/>`;
  const map =
    rect(x0, y0, box.height, grey(MOCK_DEPTH.sky)) +
    `<ellipse cx="${cx}" cy="${cy}" rx="${round(HILL.rx * box.width)}" ry="${round(HILL.ry * box.height)}" fill="${grey(MOCK_DEPTH.far)}"/>` +
    rect(nx, ny, 0.26 * box.height, grey(MOCK_DEPTH.near));
  const blurs: string[] = [];
  const sigma = 0.006 * long + row.feather * 0.02 * long;
  let plane = map;
  if (sigma >= 0.2) {
    blurs.push(`<filter id="${id}f" ${region}><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`);
    plane = `<g filter="url(#${id}f)">${map}</g>`;
  }
  if (row.invert) {
    blurs.push(`<filter id="${id}i" ${region}><feColorMatrix type="matrix" values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter>`);
    plane = `<g filter="url(#${id}i)">${plane}</g>`;
  }
  const mask = `<mask id="${id}" maskUnits="userSpaceOnUse" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}">${plane}</mask>`;
  return `<defs>${blurs.join("")}${mask}</defs><rect data-depth-row="" data-far="${round(row.far)}" x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}" fill-opacity="${opacity}" mask="url(#${id})"/>`;
}

/**
 * Show depth on the flat scene (`docs/23` §4.8): the depth map as grey, **white for near**, whole and not cut to a band: the sky over the frame, the far hill (the half
 * ellipse of `scenes.ts`) and the near band, each at the grey of its depth position, a little soft as a map of coarse cells is. Before the map is made
 * (`ready` false) the view selects nothing and is black, as the core's is.
 */
export function depthViewSvg(box: MaskRect, photo?: MaskPhoto, ready = true): string {
  const frame = photo ? { x: 0, y: 0, width: photo.width, height: photo.height } : box;
  const long = Math.max(frame.width, frame.height);
  const grey = (z: number): string => {
    const v = Math.round(255 * (1 - z));
    return `rgb(${v},${v},${v})`;
  };
  const at = (x: number, y: number): [string, string] => [round(frame.x + x * frame.width), round(frame.y + y * frame.height)];
  const [x0, y0] = at(0, 0);
  const [nx, ny] = at(0, 0.74);
  const [cx, cy] = at(HILL.cx, HILL.cy);
  const rect = (x: string, y: string, h: number, fill: string): string => `<rect x="${x}" y="${y}" width="${round(frame.width)}" height="${round(h)}" fill="${fill}"/>`;
  const map = ready
    ? rect(x0, y0, frame.height, grey(MOCK_DEPTH.sky)) +
      `<ellipse cx="${cx}" cy="${cy}" rx="${round(HILL.rx * frame.width)}" ry="${round(HILL.ry * frame.height)}" fill="${grey(MOCK_DEPTH.far)}"/>` +
      rect(nx, ny, 0.26 * frame.height, grey(MOCK_DEPTH.near))
    : "";
  const sigma = 0.006 * long;
  const filter = `<filter id="mvdepthblur" filterUnits="userSpaceOnUse" x="${round(frame.x)}" y="${round(frame.y)}" width="${round(frame.width)}" height="${round(frame.height)}"><feGaussianBlur stdDeviation="${round(sigma)}"/></filter>`;
  const content = `<rect x="${round(frame.x)}" y="${round(frame.y)}" width="${round(frame.width)}" height="${round(frame.height)}" fill="#000"/><g filter="url(#mvdepthblur)">${map}</g>`;
  const placed = photo ? `<g transform="matrix(${photo.matrix.map(round).join(" ")})">${content}</g>` : content;
  return `<g data-mask-view="depth"><defs>${filter}</defs>${placed}</g>`;
}

/** One row's gradient shape painted with `ink` (a luminance, "#fff" or "#000") at `m` and nothing outside; empty for a kind that selects nothing. */
function rowShape(row: MaskItem, index: number, ink: string, box: MaskRect, found: VisionFound, depthReady: boolean): string {
  const long = Math.max(box.width, box.height);
  const id = `mv${index}`;
  const opacity = Math.max(0, Math.min(1, row.opacity));
  if (row.kind === "brush") return brushShape(row, index, ink, box);
  if (row.kind === "background") return backgroundShape(row, index, ink, box);
  if (row.kind === "depth") return depthShape(row, index, ink, box, depthReady);
  if (row.kind === "object") return objectShape(row, index, ink, box);
  if (row.kind === "subject" || row.kind === "people") return visionShape(row, index, ink, box, found);
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
function maskContent(masks: MaskSettings, box: MaskRect, found: VisionFound, depthReady: boolean): string {
  const rows = masks.items.filter((row) => row.enabled);
  const flip = masks.invert;
  const paper = (ink: string): string => `<rect x="${round(box.x)}" y="${round(box.y)}" width="${round(box.width)}" height="${round(box.height)}" fill="${ink}"/>`;
  if (rows.length === 0) return paper(flip ? "#000" : "#fff");
  // A lone Subtract starts from the whole photo; every other list starts from nothing.
  const first = rows[0];
  const base = first?.mode === "subtract" ? "#fff" : "#000";
  const shapes = rows
    .map((row, i) => rowShape(row, i, row.mode === "subtract" ? "#000" : "#fff", box, found, depthReady))
    .join("");
  const content = paper(base) + shapes;
  // The node's Invert is the same picture with the two inks exchanged.
  return flip ? `<g filter="url(#mvinvert)">${content}</g>` : content;
}

/**
 * The masks a mask view draws (`docs/23` §5.8): the node's own, or, for a row's view, that row alone as it enters the node's fold: on whatever its eye
 * says, added whatever its mode, with its own Invert and Opacity, and no Invert of the node's. A row the node does not have is the node's own view.
 */
export function masksOfView(masks: MaskSettings, row: number | null | undefined): MaskSettings {
  const item = row === null || row === undefined ? undefined : masks.items[row];
  return item ? { invert: false, items: [{ ...item, enabled: true, mode: "add" }] } : masks;
}

/**
 * The SVG that draws a mask view over the picture at `box` (in the frame's pixels): `Overlay` tints the accent where the mask is 1,
 * `Mask only` paints the mask as grey over black. With `depth.on` and a Depth row it is the depth map as grey, white for near (`depthViewSvg`). An empty string for no view.
 */
export function maskViewSvg(
  style: MaskStyle,
  masks: MaskSettings,
  box: MaskRect,
  photo?: MaskPhoto,
  found: VisionFound = { subject: true, people: false },
  depth?: { on: boolean; ready: boolean },
): string {
  // Show depth (`MaskView.depth`): for a Depth row the picture is the depth map as grey, whatever the style; for any other row the flag means nothing.
  if (depth?.on && masks.items.length === 1 && masks.items[0]?.kind === "depth") return depthViewSvg(box, photo, depth.ready);
  // A Depth row selects nothing until the photo's depth map is made (a viewer never waits); without the flag the map is taken as made.
  const depthReady = depth?.ready ?? true;
  // The shapes are in the photo frame: drawn on a rectangle of the photo's own size and carried to the picture by its map (the turn and the crop).
  const content = photo
    ? `<g transform="matrix(${photo.matrix.map(round).join(" ")})">${maskContent(masks, { x: 0, y: 0, width: photo.width, height: photo.height }, found, depthReady)}</g>`
    : maskContent(masks, box, found, depthReady);
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
