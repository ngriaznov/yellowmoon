import type { NodeProps } from "@xyflow/react";
import { ScanFaceIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import type { KeyboardEvent } from "react";
import type { NodeId } from "../../api/types";
import { Input } from "@/components/glass/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import { cardName, chipText } from "../../state/graphOps";
import { isCardItem, nodeOf } from "../../state/settingsGraph";
import type { CardItem } from "../../state/settingsGraph";
import { Port } from "./Port";
import { useGraphUi } from "./graphUi";

/** The name above a card is cut at about ten letters (`docs/23` §7.2); the whole name is its tooltip. */
export const NAME_CUT = 10;

/** A name as it is drawn above a card: cut at ten letters, with an ellipsis. */
export function shortName(name: string): string {
  const letters = Array.from(name);
  return letters.length <= NAME_CUT ? name : `${letters.slice(0, NAME_CUT - 1).join("").trimEnd()}…`;
}

/** What a card's footer tells about its masks: the AI kinds (Object, Subject, People, Background) and the others. */
const AI_KINDS: ReadonlySet<string> = new Set(["object", "subject", "people", "background"]);

export function maskGlyphsOf(node: Pick<CardItem, "masks">): { ai: boolean; plain: boolean } {
  let ai = false;
  let plain = false;
  for (const row of node.masks.items) {
    if (!row.enabled) continue;
    if (AI_KINDS.has(row.kind)) ai = true;
    else plain = true;
  }
  return { ai, plain };
}

/** The sparkle of "AI mask", from the mockup. */
function SparkleGlyph() {
  return (
    <svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />
      <path d="M20 3v4" />
      <path d="M22 5h-4" />
    </svg>
  );
}

/** The half-filled disc of "Mask". */
function HalfDiscGlyph() {
  return (
    <svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" />
    </svg>
  );
}

/** The circle with a slash of a bypassed node. */
function SlashGlyph() {
  return (
    <svg aria-hidden="true" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="8" />
      <path d="m6 18 12-12" />
    </svg>
  );
}

/** A glyph of the footer with the tooltip that names it. */
function Tip({ text, glyph, className, children }: { text: string; glyph: "ai" | "mask" | "bypass" | "face"; className?: string; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={className} data-glyph={glyph}>
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent>{text}</TooltipContent>
    </Tooltip>
  );
}

/** The name above a card: the name cut at ten letters, double-click to rename; while it is renamed a text box. */
export function CardName({ id, name }: { id: NodeId; name: string }) {
  const ui = useGraphUi();
  const editing = ui.renaming === id;
  const finished = useRef(false);
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (!editing) return;
    finished.current = false;
    field.current?.focus();
    field.current?.select();
  }, [editing]);
  if (editing) {
    // Return and Esc give the focus back to the card; leaving the box (a click elsewhere) leaves it where the person put it.
    const finish = (value: string | null, restoreFocus: boolean): void => {
      if (finished.current) return;
      finished.current = true;
      ui.onRename(id, value, restoreFocus);
    };
    const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
      // The name is the text box's own until it is done: no shortcut of the screen sees these keys.
      event.stopPropagation();
      if (event.key === "Enter") {
        event.preventDefault();
        finish(event.currentTarget.value, true);
      } else if (event.key === "Escape") {
        event.preventDefault();
        finish(null, true);
      }
    };
    return (
      <Input
        ref={field}
        size="sm"
        className="ym-gnode__rename nodrag nopan"
        aria-label="Node name"
        maxLength={32}
        defaultValue={name}
        spellCheck={false}
        onKeyDown={onKeyDown}
        onBlur={(event) => finish(event.currentTarget.value, false)}
        onPointerDown={(event) => event.stopPropagation()}
      />
    );
  }
  const cut = shortName(name) !== name;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="ym-gnode__name" data-node-name onDoubleClick={() => ui.onStartRename(id)}>
          {shortName(name)}
        </span>
      </TooltipTrigger>
      <TooltipContent>{cut ? name : "Double-click to rename"}</TooltipContent>
    </Tooltip>
  );
}

/**
 * A corrector's card (`docs/23` §7.2): 150 × 112 px with the name above it, a thumbnail band of 81 px, and a footer of 30 px with the
 * number chip at the left and the mask glyphs at the right; the ports on its sides. Selected: a 2 px yellow ring. Bypassed: the picture
 * at half strength and a slash glyph. Dead (not on a path to the Output): the whole card at 45 % with the tooltip "Not connected to the
 * output.". It is one button (`aria-pressed` for the selection) and the graph's one Tab stop is the selected card.
 */
export function NodeCard({ data }: NodeProps) {
  const ui = useGraphUi();
  const id = (data as { id: NodeId }).id;
  const node = nodeOf(ui.settings, id);
  if (!isCardItem(node)) return null;
  const state = ui.states.get(node.id as number);
  const dead = state?.live === false;
  const selected = ui.selected === id;
  const glyphs = maskGlyphsOf(node);
  const thumbnail = ui.thumbnailOf(id);
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
  return (
    <div className="ym-gnode" data-selected={selected || undefined} data-quiet={(selected && ui.linkSelected) || undefined} data-bypass={node.bypass || undefined} data-dead={dead || undefined}>
      <CardName id={id} name={node.name} />
      <OptionalTooltip text={dead ? "Not connected to the output." : null}>
        <TooltipTrigger asChild>
          <div
            className="ym-gnode__box"
            role="button"
            tabIndex={selected ? 0 : -1}
            aria-pressed={selected}
            aria-label={cardName(ui.settings, id, ui.states)}
            data-node-card={String(id)}
            onClick={() => ui.onSelect(id)}
            onKeyDown={onKeyDown}
          >
            <div className="ym-gnode__thumb">
              {thumbnail ? <img src={thumbnail} alt="" draggable={false} /> : null}
              {node.bypass ? <span className="ym-gnode__slash" /> : null}
            </div>
            <div className="ym-gnode__foot">
              <span className="ym-gnode__num">{chipText(state?.number ?? 0)}</span>
              <span className="ym-gnode__tools">
                {node.kind === "face_refinement" ? (
                  <Tip text="Face Refinement" glyph="face" className="ym-gnode__face">
                    <ScanFaceIcon aria-hidden="true" width={13} height={13} />
                  </Tip>
                ) : null}
                {glyphs.ai ? (
                  <Tip text="AI mask" glyph="ai" className="ym-gnode__mask">
                    <SparkleGlyph />
                  </Tip>
                ) : null}
                {glyphs.plain ? (
                  <Tip text="Mask" glyph="mask" className="ym-gnode__mask">
                    <HalfDiscGlyph />
                  </Tip>
                ) : null}
                {node.bypass ? (
                  <Tip text="Bypassed" glyph="bypass">
                    <SlashGlyph />
                  </Tip>
                ) : null}
              </span>
            </div>
          </div>
        </TooltipTrigger>
      </OptionalTooltip>
      <span className="ym-gnode__ring" aria-hidden="true" />
      <Port node={id} port="picture" side="in" on="card" />
      <Port node={id} port="picture" side="out" on="card" />
      <Port node={id} port="key" side="in" on="card" />
      <Port node={id} port="key" side="out" on="card" />
    </div>
  );
}
