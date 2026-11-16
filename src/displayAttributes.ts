import type { DisplayOptions } from "./api/types";

/**
 * The attribute of `<html>` for each display option of the system (`docs/28` A-P12). The CSS answers these next to the media queries
 * (`src/styles/glass.css`, `global.css`); `src/styles/displayOptions.node.test.ts` holds the two to the same three names. No DOM here, so
 * that a node test can import it.
 */
export const DISPLAY_ATTRIBUTES = {
  reduce_transparency: "data-reduce-transparency",
  reduce_motion: "data-reduce-motion",
  increase_contrast: "data-increase-contrast",
} as const satisfies Record<keyof DisplayOptions, string>;
