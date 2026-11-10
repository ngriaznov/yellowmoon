/**
 * The ports of the graph's nodes as the page needs them (`docs/25-mixers.md` §2.2): the type each carries, the written order, the numbered
 * inputs of a mixer and the channels of a Splitter and a Combiner. Pure, so the layout, the operations and the panels share one reading.
 */
import type { NodeKind, Port } from "../api/types";

/** Most picture inputs a mixer has (`docs/25` §2.1). */
export const MIXER_INPUTS = 8;

/** The numbered inputs of a mixer, `picture.1` to `picture.8`. */
export const MIXER_PORTS: readonly Port[] = ["picture.1", "picture.2", "picture.3", "picture.4", "picture.5", "picture.6", "picture.7", "picture.8"];

/** The channel ports of a Splitter's outputs and a Combiner's inputs, in Resolve's order. */
export const CHANNEL_PORTS: readonly Port[] = ["red", "green", "blue"];

/** True for a Parallel and a Layer. */
export const isMixerKind = (kind: NodeKind): kind is "parallel" | "layer" => kind === "parallel" || kind === "layer";

/** True for the four kinds of `docs/25`: they have a glyph card, no number, no masks and no key ports. */
export const isGlyphKind = (kind: NodeKind): kind is "parallel" | "layer" | "splitter" | "combiner" =>
  kind === "parallel" || kind === "layer" || kind === "splitter" || kind === "combiner";

/** The input `picture.k`: its number k, or null for any other port. */
export function inputNumber(port: Port): number | null {
  const match = /^picture\.([1-8])$/.exec(port);
  return match ? Number(match[1]) : null;
}

/** The port `picture.k` of a mixer's input k (1 to 8). */
export const mixerPort = (k: number): Port => MIXER_PORTS[k - 1] as Port;

/** What a port carries: every port but `key` carries a picture (`docs/25` §2.2). A link joins two ports of the same type. */
export const portType = (port: Port): "picture" | "key" => (port === "key" ? "key" : "picture");

/** A port's place in the order links are written in: `picture`, `key`, `picture.1` to `picture.8`, then `red`, `green`, `blue`. */
export function portRank(port: Port): number {
  if (port === "picture") return 0;
  if (port === "key") return 1;
  const n = inputNumber(port);
  if (n !== null) return 1 + n;
  return port === "red" ? 20 : port === "green" ? 21 : 22;
}

/** The first picture input of a kind: where the node's picture comes in (a mixer's `picture.1`, a Combiner's `red`). */
export function firstInputPort(kind: NodeKind): Port {
  if (isMixerKind(kind)) return "picture.1";
  return kind === "combiner" ? "red" : "picture";
}

/** The letter a channel port shows inside a glyph card. */
export const CHANNEL_LETTER: Readonly<Record<string, string>> = { red: "R", green: "G", blue: "B" };

/** The tint of each channel port (`design/mixers`, option C: tinted ports and letters). */
export const CHANNEL_TINT: Readonly<Record<string, string>> = { red: "#ff7a6e", green: "#62d68f", blue: "#6fadff" };

/** How a link into a port or out of one is named in a sentence: "input 2", "red", "a new input". */
export function portWords(port: Port): string {
  const n = inputNumber(port);
  return n !== null ? `input ${n}` : port;
}
