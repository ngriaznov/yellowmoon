/**
 * The Colour Uniform node on the mock's flat-shape photos (`docs/32` §3). The mock has no pipeline and draws no masks: the node acts on the
 * whole scene, as a node with no mask does. Each of the four flat colours is taken to the approximate LogC4 pixel of the colour warp's
 * fixture (`warp.ts`), moved by the TypeScript port of the node (`components/uniformMath.ts`) and brought back. The same approximate pixel
 * gives `sample_chroma` its `tone`, so a pick lands on the colour it was made from. Nothing here is colour science.
 */
import type { UniformAmounts, TargetValue } from "../api/types";
import { applyUniform, fromStored, toneOf } from "../components/uniformMath";
import { cardStates } from "../state/graphOps";
import type { DevelopSettings } from "../api/types";
import type { ColorUniformItem } from "../state/settingsGraph";
import { hexToLogC4, logC4ToHex } from "./warp";
import type { Scene } from "./scenes";

const KEYS = ["sky", "far", "near", "sun"] as const;

/** The scene with its four colours through the node. No target, or three amounts at 0, leaves it as it is, to the byte. */
export function uniformScene(scene: Scene, target: TargetValue | null, compress: UniformAmounts): Scene {
  if (!target || (compress.hue === 0 && compress.saturation === 0 && compress.lightness === 0)) return scene;
  const plane = fromStored(target);
  const out: Scene = { ...scene };
  for (const key of KEYS) out[key] = logC4ToHex(applyUniform(hexToLogC4(scene[key]), plane, compress));
  return out;
}

/** The tone (the mean of the three LogC4 codes) of an sRGB `#RRGGBB`, as the mock reads it. */
export const hexTone = (hex: string): number => toneOf(hexToLogC4(hex));

/** The scene through every live Colour Uniform node that is not bypassed, in the order of their numbers in the plan. */
export function sceneThroughUniforms(scene: Scene, settings: DevelopSettings): Scene {
  const states = cardStates(settings);
  return settings.nodes.items
    .filter((item): item is ColorUniformItem => item.kind === "color_uniform" && !item.bypass && states.get(item.id as number)?.live !== false)
    .sort((a, b) => (states.get(a.id as number)?.number ?? 0) - (states.get(b.id as number)?.number ?? 0))
    .reduce((out, item) => uniformScene(out, item.target, item.compress), scene);
}
