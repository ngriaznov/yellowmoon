import { useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import { ChevronsRightIcon, LayersIcon } from "lucide-react";
import { useEffect } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import type { NodeId } from "../../api/types";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import { cardName } from "../../state/graphOps";
import { CHANNEL_LETTER, CHANNEL_TINT } from "../../state/ports";
import { isGlyphItem, nodeOf } from "../../state/settingsGraph";
import { CardName } from "./NodeCard";
import { Port } from "./Port";
import { glyphGeometry } from "./graphLayout";
import { useGraphUi } from "./graphUi";

/** The fan of a Splitter (one line in, three out) and a Combiner (three in, one out), `design/mixers`. */
function FanGlyph({ kind }: { kind: "splitter" | "combiner" }) {
  const split = kind === "splitter";
  return (
    <svg
      className="ym-glyph"
      aria-hidden="true"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {split ? (
        <>
          <path d="M3 12h5" />
          <path d="M8 12c3 0 3.5-7 8-7h5" />
          <path d="M8 12h13" />
          <path d="M8 12c3 0 3.5 7 8 7h5" />
        </>
      ) : (
        <>
          <path d="M21 12h-5" />
          <path d="M16 12c-3 0-3.5-7-8-7H3" />
          <path d="M16 12H3" />
          <path d="M16 12c-3 0-3.5 7-8 7H3" />
        </>
      )}
    </svg>
  );
}

/** The kind's glyph: the menu's icons for the mixers, the fans for the channel nodes. */
export function KindGlyph({ kind }: { kind: "parallel" | "layer" | "splitter" | "combiner" }) {
  if (kind === "parallel") return <ChevronsRightIcon className="ym-glyph" aria-hidden="true" width={20} height={20} />;
  if (kind === "layer") return <LayersIcon className="ym-glyph" aria-hidden="true" width={20} height={20} />;
  return <FanGlyph kind={kind} />;
}

/**
 * The card of a Parallel, a Layer, a Splitter or a Combiner (`docs/25` §5.2): 56 px wide, 64 px high with two ports and 24 px more for each
 * further one; a rounded box with the kind's glyph, the name above it, **no thumbnail, no number chip, no footer**. Numbered inputs are
 * stacked on the left in port order with the "+" under the last while a mixer has fewer than 8; a Splitter's outputs and a Combiner's inputs are
 * red, green and blue, tinted, with the letter beside each inside the card. Selected: the 2 px yellow ring. Bypassed: the glyph at half
 * strength and the slash. Dead: the whole card at 45 % with "Not connected to the output.". One button, as a corrector's card is.
 */
export function GlyphCard({ data }: NodeProps) {
  const ui = useGraphUi();
  const update = useUpdateNodeInternals();
  const id = (data as { id: NodeId }).id;
  const node = nodeOf(ui.settings, id);
  const geometry = glyphGeometry(ui.settings, id);
  // The ports move with the inputs: xyflow measures the handles again.
  const signature = geometry
    ? [geometry.h, ...geometry.inputs.map((s) => `${s.port}${s.plus ? "+" : ""}`), ...geometry.outputs.map((s) => s.port)].join(",")
    : "";
  useEffect(() => {
    update(String(id));
  }, [update, id, signature]);
  if (!isGlyphItem(node) || !geometry) return null;
  const dead = ui.states.get(node.id as number)?.live === false;
  const selected = ui.selected === id;
  const bypassed = "bypass" in node && node.bypass;
  const channels = node.kind === "splitter" || node.kind === "combiner";
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.altKey || event.metaKey || event.ctrlKey) return;
    const direction = { ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down" }[event.key] as "left" | "right" | "up" | "down" | undefined;
    if (direction) {
      event.preventDefault();
      ui.onNavigate(id, direction);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      ui.onSelect(id);
    }
  };
  const size: CSSProperties = { width: geometry.w, height: geometry.h };
  return (
    <div
      className="ym-gnode ym-gnode--glyph"
      style={size}
      data-selected={selected || undefined}
      data-quiet={(selected && ui.linkSelected) || undefined}
      data-bypass={bypassed || undefined}
      data-dead={dead || undefined}
      data-glyph={node.kind}
    >
      <CardName id={id} name={node.name} />
      <OptionalTooltip text={dead ? "Not connected to the output." : null}>
        <TooltipTrigger asChild>
          <div
            className="ym-gnode__box ym-gnode__glyphbox"
            style={size}
            role="button"
            tabIndex={selected ? 0 : -1}
            aria-pressed={selected}
            aria-label={cardName(ui.settings, id, ui.states)}
            data-node-card={String(id)}
            onClick={() => ui.onSelect(id)}
            onKeyDown={onKeyDown}
          >
            <KindGlyph kind={node.kind} />
            {channels
              ? (node.kind === "splitter" ? geometry.outputs : geometry.inputs).map((spot) => (
                  <span
                    key={spot.port}
                    className="ym-gnode__letter"
                    data-side={node.kind === "splitter" ? "right" : "left"}
                    style={{ top: spot.y - 7, "--ch": CHANNEL_TINT[spot.port] } as CSSProperties}
                    aria-hidden="true"
                  >
                    {CHANNEL_LETTER[spot.port]}
                  </span>
                ))
              : null}
            {bypassed ? <span className="ym-gnode__slash" /> : null}
          </div>
        </TooltipTrigger>
      </OptionalTooltip>
      <span className="ym-gnode__ring" aria-hidden="true" />
      {geometry.inputs.map((spot) => (
        <Port key={`in:${spot.port}${spot.plus ? "+" : ""}`} node={id} port={spot.port} side="in" on="glyph" y={spot.y} plus={spot.plus} />
      ))}
      {geometry.outputs.map((spot) => (
        <Port key={`out:${spot.port}`} node={id} port={spot.port} side="out" on="glyph" y={spot.y} />
      ))}
    </div>
  );
}
