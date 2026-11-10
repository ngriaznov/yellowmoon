/**
 * What the four kinds of `docs/25` do to the mock's flat scenes (four colours: sky, far hills, near ground, sun). The core runs §3 on the
 * pictures' pixels (`ym_color`); the mock has four colours per scene, so this is a small CPU approximation of the same formulas on them,
 * enough for the page's states, the e2e tests and the screenshots, not a copy of the maths:
 *
 * - **Parallel**: `out = base + Σ (inᵢ − base)` per channel, on the colours' 0 to 255 values standing in for LogC4 codes (the base is the
 *   immediate dominator, `baseOf`; an input that is the base adds nothing). A bypassed Parallel gives its base.
 * - **Layer**: input 1 at the bottom, a higher number on top; each layer is blended over the stack in "linear" light (the colours at a
 *   gamma of 2.2) by the seven modes and laid on with its alpha, `acc + α (B(acc, layer) − acc)`. The mock draws no masks, so α is the
 *   Opacity (`docs/25` §5.7). A bypassed Layer gives input 1.
 * - **Splitter and Combiner**: `(r, r, r)`, `(g, g, g)` and `(b, b, b)` out, and `(red.r, green.g, blue.b)` back in.
 *
 * Correctors and Film nodes are graded by the caller (`apply`), so the same walk serves the frame and the node thumbnails.
 */
import type { BlendMode, DevelopSettings, NodeRef } from "../api/types";
import { baseOf, cardStates, inputsOf } from "../state/graphOps";
import { isGlyphItem, nodeOf } from "../state/settingsGraph";
import type { CardItem } from "../state/settingsGraph";
import type { Scene } from "./scenes";

const KEYS = ["sky", "far", "near", "sun"] as const;
type Rgb = [number, number, number];

const GAMMA = 2.2;
const toRgb = (hex: string): Rgb => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)) as Rgb;
const toHex = (c: readonly number[]): string =>
  `#${c
    .map((x) =>
      Math.round(Math.min(255, Math.max(0, x)))
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
const lin = (x: number): number => (Math.max(0, x) / 255) ** GAMMA;
const enc = (x: number): number => 255 * Math.max(0, x) ** (1 / GAMMA);

/** A blend mode on linear values, `a` the stack and `b` the layer (`docs/25` §3.3). */
export function blend(mode: BlendMode, a: number, b: number): number {
  switch (mode) {
    case "normal":
      return b;
    case "add":
      return a + b;
    case "subtract":
      return a - b;
    case "multiply":
      return a * b;
    case "lighten":
      return Math.max(a, b);
    case "darken":
      return Math.min(a, b);
    case "difference":
      return Math.abs(a - b);
  }
}

/** True when the graph has a Parallel, a Layer, a Splitter or a Combiner that the Output needs (a dead one changes nothing). */
export function hasMixers(settings: DevelopSettings): boolean {
  const states = cardStates(settings);
  return settings.nodes.items.some((item) => isGlyphItem(item) && states.get(item.id as number)?.live === true);
}

/** The output that feeds the Output node. */
export const outputFeeder = (settings: DevelopSettings): NodeRef =>
  settings.nodes.links.find((l) => l.to.node === "output")?.from ?? { node: 1, port: "picture" };

/** Maps each of the four colours' channels. */
function perColour(scene: Scene, f: (c: Rgb, key: (typeof KEYS)[number]) => Rgb): Scene {
  const out: Scene = { ...scene };
  for (const key of KEYS) out[key] = toHex(f(toRgb(scene[key]), key));
  return out;
}

/** The scene at one output of a node, given the picture the Input gives and what a corrector or a Film node does to a scene. */
export function sceneAt(settings: DevelopSettings, input: Scene, at: NodeRef, apply: (node: CardItem, scene: Scene) => Scene): Scene {
  const memo = new Map<string, Scene>();
  const links = settings.nodes.links;
  const source = (node: number, port: NodeRef["port"]): NodeRef =>
    links.find((l) => l.to.node === node && l.to.port === port)?.from ?? { node: "input", port: "picture" };
  const value = (ref: NodeRef, depth = 0): Scene => {
    if (ref.node === "input" || depth > 80) return input;
    const key = `${ref.node}:${ref.port}`;
    const known = memo.get(key);
    if (known) return known;
    const item = nodeOf(settings, ref.node);
    let out: Scene = input;
    if (item?.kind === "corrector" || item?.kind === "film") {
      out = apply(item, value(source(item.id as number, "picture"), depth + 1));
    } else if (item?.kind === "parallel") {
      const rows = inputsOf(settings, item.id);
      const ins = rows.map((row) => ({
        from: row.link?.from ?? ({ node: "input", port: "picture" } as NodeRef),
        scene: value(row.link?.from ?? { node: "input", port: "picture" }, depth + 1),
      }));
      const baseRef = baseOf(settings, item.id);
      const base = value(baseRef, depth + 1);
      if (item.bypass) {
        out = base;
      } else {
        out = perColour(base, (b, k) => {
          const acc: Rgb = [...b];
          for (const row of ins) {
            if (row.from.node === baseRef.node && row.from.port === baseRef.port) continue;
            const c = toRgb(row.scene[k]);
            for (let i = 0; i < 3; i++) acc[i] = (acc[i] as number) + ((c[i] as number) - (b[i] as number));
          }
          return acc;
        });
      }
    } else if (item?.kind === "layer") {
      const rows = inputsOf(settings, item.id);
      const layers = rows.map((row) => value(row.link?.from ?? { node: "input", port: "picture" }, depth + 1));
      const bottom = layers[0] ?? input;
      if (item.bypass || layers.length < 2 || item.opacity === 0) {
        out = bottom;
      } else {
        out = perColour(bottom, (_c, k) => {
          let acc = toRgb(bottom[k]).map(lin);
          for (const layer of layers.slice(1)) {
            const b = toRgb(layer[k]).map(lin);
            acc = acc.map((a, i) => {
              const c = blend(item.mode, a, b[i] as number);
              return item.opacity === 1 ? c : a + item.opacity * (c - a);
            });
          }
          return acc.map(enc) as Rgb;
        });
      }
    } else if (item?.kind === "splitter") {
      const picture = value(source(item.id as number, "picture"), depth + 1);
      const channel = ref.port === "red" ? 0 : ref.port === "green" ? 1 : 2;
      out = perColour(picture, (c) => [c[channel] as number, c[channel] as number, c[channel] as number]);
    } else if (item?.kind === "combiner") {
      const [r, g, b] = (["red", "green", "blue"] as const).map((port) => value(source(item.id as number, port), depth + 1));
      out = perColour(input, (_c, k) => [toRgb((r as Scene)[k])[0], toRgb((g as Scene)[k])[1], toRgb((b as Scene)[k])[2]]);
    }
    memo.set(key, out);
    return out;
  };
  return value(at);
}

/**
 * The scene a panel signal reads at a mixer, a Splitter or a Combiner (`docs/25` §5.7): the picture after the node; for a Splitter, the
 * picture on the link into it. Null for a Splitter with nothing linked in. `apply` is what a corrector or a Film node does to a scene.
 */
export function sceneAfter(settings: DevelopSettings, input: Scene, id: number, apply: (node: CardItem, scene: Scene) => Scene): Scene | null {
  let at: NodeRef = { node: id, port: "picture" };
  if (nodeOf(settings, id)?.kind === "splitter") {
    const into = settings.nodes.links.find((l) => l.to.node === id && l.to.port === "picture");
    if (!into) return null;
    at = into.from;
  }
  return sceneAt(settings, input, at, apply);
}
