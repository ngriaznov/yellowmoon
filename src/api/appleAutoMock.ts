/**
 * The mock's copies of the core's Apple Auto constants (`ym_core::apple_auto`, `docs/19` sections 2, 4 and 5): the default
 * instructions, their limit and digest, and the table. They live apart from `src/api.mock.ts` so that `src/appleAutoMock.node.test.ts`
 * can read them without a browser and check them against the Rust file.
 */
import type { AppleHighlightColor, AppleShadowColor, ColorBalancePoint } from "./types";

/** The default instructions for Apple's model (`ym_core::apple_auto::DEFAULT_INSTRUCTIONS`, `docs/19` section 4). */
export const MOCK_DEFAULT_INSTRUCTIONS =
  "You are a master colorist grading a raw, neutral camera file inside a RAW processor built on an ARRI color pipeline. Your goal is to transform this flat starting point into the best finished, professional photograph the source, light, subject and scene allow, honoring ARRI's signature color science. This is the base look. Add no style: the shadow colour and highlight colour must be none (the light is already correct), and no mood.\n1. Analyze & Respect the Scene\nPreserve the natural logic of the light. Night scenes stay dark; snow and high-key scenes stay bright; silhouettes stay dark; candlelight and golden hour stay warm. Keep skin tones natural, organic and true to life.\n2. Execute a Comprehensive Grade\nDo not rely on a single adjustment; a finished grade usually changes several controls, each chosen for what this picture needs:\n* Exposure & Tone: set the subject's brightness to guide the eye; brighten a subject that is too dark, hold one that is well lit, never darken a dark photo, and keep night scenes and silhouettes dark. Bring bright skies and whites down for smooth ARRI-style highlight roll-off. Bright sky, clouds, white clothes or lights: highlights darker for roll-off. Shadows brighter only where dark areas hide real detail; leave them when the shadows already show detail. Blacks deeper only if the picture looks hazy or milky; if the blacks are already deep and clean, leave them.\n* Contrast: deliberate dynamic range that fits the light: more for flat or hazy light, less for harsh light, with rich midtone detail.\n* Color Balance & Density: neutral whites, accurate warmth and tint, fixing casts. Saturation true and dense, neither boosted nor washed out.";

/** The longest instructions the model is given, in characters (`INSTRUCTIONS_MAX_CHARS`). */
export const MOCK_INSTRUCTIONS_MAX_CHARS = 2000;

/** The first 12 hex digits of the SHA-256 of the default instructions (what the core reports). */
export const MOCK_DEFAULT_INSTRUCTIONS_SHA256 = "932b96907b63";

/** The table of `docs/19` section 5 (`ym_core::apple_auto::TABLE`, table version 2), by step. */
export const APPLE_TABLE = {
  exposure_ev: { [-3]: -1, [-2]: -0.6, [-1]: -0.3, 0: 0, 1: 0.3, 2: 0.6, 3: 1 } as Record<number, number>,
  highlights: { [-2]: -40, [-1]: -20, 0: 0, 1: 20, 2: 40 } as Record<number, number>,
  warmer_mired: { [-2]: -20, [-1]: -10, 0: 0, 1: 10, 2: 20 } as Record<number, number>,
  tint: { [-2]: -10, [-1]: -5, 0: 0, 1: 5, 2: 10 } as Record<number, number>,
  percent: { [-2]: -30, [-1]: -15, 0: 0, 1: 15, 2: 30 } as Record<number, number>,
  strength_radius: { 1: 0.3, 2: 0.55, 3: 0.8 } as Record<number, number>,
  shadow_directions: {
    none: [0, 0],
    teal: [-0.866, -0.5],
    blue: [-0.5, -0.866],
    green: [-0.5, 0.866],
    amber: [0.866, 0.5],
    magenta: [0.5, -0.866],
  } as Record<AppleShadowColor, ColorBalancePoint>,
  highlight_directions: {
    none: [0, 0],
    warm: [0.866, 0.5],
    gold: [0.5, 0.866],
    cool: [-0.5, -0.866],
    pink: [0.866, -0.5],
  } as Record<AppleHighlightColor, ColorBalancePoint>,
};
