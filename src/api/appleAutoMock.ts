/**
 * The mock's copies of the core's Apple Auto constants (`ym_core::apple_auto`, `docs/19` sections 2, 4 and 5): the default
 * instructions, their limit and digest, and the table. They live apart from `src/api.mock.ts` so that `src/appleAutoMock.node.test.ts`
 * can read them without a browser and check them against the Rust file.
 */
import type { AppleHighlightColor, AppleShadowColor, ColorBalancePoint } from "./types";

/** The default instructions for Apple's model (`ym_core::apple_auto::DEFAULT_INSTRUCTIONS`, `docs/19` section 4). */
export const MOCK_DEFAULT_INSTRUCTIONS =
  "You are a professional colourist grading raw photos. The picture you see is the camera's neutral starting point, not a finished photo. Grade it into a finished, professional photograph, and give it a cinematic look when the scene invites it: dusk and night streets, fog and rain, dramatic or directional light, people in a mood. Work like a colourist: set the brightness of the subject, shape the highlights, the shadows and the black level, set the contrast, the warmth and tint of the light and the strength of the colours, and colour the shadows and the highlights when it serves the mood. A professional grade usually changes several of these. Respect the scene: a night scene stays dark, snow stays bright, a silhouette stays dark, candlelight and golden hour stay warm, and skin stays natural.";

/** The longest instructions the model is given, in characters (`INSTRUCTIONS_MAX_CHARS`). */
export const MOCK_INSTRUCTIONS_MAX_CHARS = 2000;

/** The first 12 hex digits of the SHA-256 of the default instructions (what the core reports). */
export const MOCK_DEFAULT_INSTRUCTIONS_SHA256 = "6a8a034fb24b";

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
