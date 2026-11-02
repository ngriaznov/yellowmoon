/**
 * The geometry of the node view, layout A (`docs/23` §7.1): the header, the photo cell on top, a 10 px divider and the graph editor
 * under it, the panel at the right. Everything is a function of the window's height and of one number the person can change, the
 * editor's share of the height under the header: 45 % to start with, 66 % while Expand is on, dragged or nudged anywhere between 25 %
 * and 70 % with the photo keeping at least 200 px. The share, not the pixels, is what the window remembers (it is window state, never
 * saved), so a taller window keeps the proportion.
 *
 * Pure functions, tested without a screen (`layout.test.ts`); `src/styles/nodeview.node.test.ts` holds the constants to the CSS tokens
 * they repeat.
 */

/** The header's height, `--header-h`. */
export const HEADER_H = 75;
/** The window is never shorter than this, `--min-window-h`: a shorter window scrolls nothing and the screen keeps this height. */
export const MIN_WINDOW_H = 760;
/** The divider's height. */
export const DIVIDER_H = 10;
/** The photo cell keeps at least this much height, whatever the divider does. */
export const MIN_PHOTO_H = 200;

/** The editor's share of the height under the header: at the start, with Expand on, and the bounds of the divider. */
export const DEFAULT_FRACTION = 0.45;
export const EXPANDED_FRACTION = 0.66;
export const MIN_FRACTION = 0.25;
export const MAX_FRACTION = 0.7;

/** What an arrow key moves the divider by, and with ⇧. */
export const NUDGE_PX = 10;
export const NUDGE_BIG_PX = 50;

/** The sizes of one layout, in CSS pixels. */
export interface NodeLayout {
  /** The height under the header: the photo row, the divider and the editor share it. */
  area: number;
  /** The editor's height, the 44 px toolbar and the canvas. */
  editor: number;
  /** The photo row's height, which the 3:2 cell fills when the column is wide enough. */
  viewer: number;
  /** The least and the most the editor may be. */
  min: number;
  max: number;
}

/** The height under the header for a window this tall. */
export const areaHeight = (windowHeight: number): number => Math.max(windowHeight, MIN_WINDOW_H) - HEADER_H;

/** The bounds of the editor's height: 25 % at least, and 70 % at most with the photo keeping its 200 px. */
export function editorBounds(area: number): { min: number; max: number } {
  const min = Math.round(MIN_FRACTION * area);
  const max = Math.max(min, Math.min(Math.round(MAX_FRACTION * area), area - DIVIDER_H - MIN_PHOTO_H));
  return { min, max };
}

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

/** The layout for a window this tall with the editor at this share of the height under the header (kept inside the bounds). */
export function layoutOf(windowHeight: number, fraction: number): NodeLayout {
  const area = areaHeight(windowHeight);
  const { min, max } = editorBounds(area);
  const editor = clamp(Math.round(fraction * area), min, max);
  return { area, editor, viewer: area - DIVIDER_H - editor, min, max };
}

/** The share an editor of this height is of the height under the header (what the window keeps when the divider is dragged). */
export const fractionOf = (editor: number, area: number): number => editor / area;
