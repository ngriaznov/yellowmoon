/**
 * The panels of the four kinds of `docs/25` (§5.5), each with one tab: a Parallel's **Inputs** (what it does, where the inputs start from and
 * the inputs in port order), a Layer's **Blend** (Mode, Opacity and the Inputs card, from the top of the stack, with a drag handle on each
 * row), a Splitter's and a Combiner's **Channels** (what they do and where each channel goes or comes from). They have no masks and no
 * histogram. Every control writes the selected node as one patch; the rows' moves and removals are graph operations (one patch, one undo step).
 */
import { GripVerticalIcon, LockIcon } from "lucide-react";
import { useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent as ReactPointerEvent, ReactNode } from "react";
import { Item } from "@/components/glass/item";
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger } from "@/components/glass/context-menu";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/glass/select";
import type { BlendMode, DevelopSettings, NodeLink, NodeRef, PhotoSummary, ResolvedSettings } from "../../api/types";
import { Card, Field, formats, ResetLink, RowNote, Slider } from "../../components";
import { useAppState } from "../../state/context";
import { baseOf, cardStates, chipText, describeEnd, inputsOf, joinWords } from "../../state/graphOps";
import type { CardState, InputRow } from "../../state/graphOps";
import { CHANNEL_LETTER, CHANNEL_PORTS, CHANNEL_TINT, inputNumber } from "../../state/ports";
import { isGlyphItem, nodeOf, patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { KindGlyph } from "./GlyphCard";
import type { NodeGraphActions } from "./useNodeGraph";

/** The seven blend modes in the order of `docs/25` §3.3 and what the menu calls them (names only). */
export const BLEND_MODES: ReadonlyArray<{ value: BlendMode; label: string }> = [
  { value: "normal", label: "Normal" },
  { value: "add", label: "Add" },
  { value: "subtract", label: "Subtract" },
  { value: "multiply", label: "Multiply" },
  { value: "lighten", label: "Lighten" },
  { value: "darken", label: "Darken" },
  { value: "difference", label: "Difference" },
];

export interface MixerTabProps {
  store: AppStore;
  photo: PhotoSummary;
  rs: ResolvedSettings;
  /** The node the panel is on. */
  node: number;
  /** What the rows' menus and drags do (`useNodeGraph`). */
  graph: NodeGraphActions;
}

/** The name of the node that feeds a row, as the row shows it: "04 Sky" for a numbered node, the name for the others, "Input" for the Input. */
function feederName(settings: DevelopSettings, from: NodeRef, states: ReadonlyMap<number, CardState>): string {
  if (from.node === "input") return "Input";
  const item = nodeOf(settings, from.node);
  if (!item || item.kind === "output" || item.kind === "input") return "";
  if (isGlyphItem(item)) return item.kind === "splitter" ? `${item.name}, ${from.port}` : item.name;
  return `${chipText(states.get(item.id as number)?.number ?? 0)} ${item.name}`;
}

/** The picture of a row: the feeding node's thumbnail, or its glyph for a mixer, a Splitter output and a Combiner, and a lock for the Input. */
function RowPicture({ settings, from, pictures }: { settings: DevelopSettings; from: NodeRef; pictures: Readonly<Record<number, string>> | undefined }) {
  const item = from.node === "input" ? undefined : nodeOf(settings, from.node);
  const url = typeof from.node === "number" ? pictures?.[from.node] : undefined;
  return (
    <span className="ym-input-row__thumb" data-glyph={isGlyphItem(item) || from.node === "input" || !url ? "" : undefined}>
      {isGlyphItem(item) ? (
        <KindGlyph kind={item.kind} />
      ) : from.node === "input" ? (
        <LockIcon aria-hidden="true" />
      ) : url ? (
        <img src={url} alt="" draggable={false} />
      ) : null}
    </span>
  );
}

interface InputRowsProps extends MixerTabProps {
  /** A Layer's rows have a drag handle, "On top" and "Below", and Move up and Move down. A Parallel's list its inputs 1 to N and only Remove input. */
  stack: boolean;
}

/**
 * The Inputs card's rows (`docs/25` §5.5). A Layer lists them from the top of the stack (input N) to the bottom (input 1); a row has a drag
 * handle, the feeding node's picture, the input's number, its name and "On top" on the first row and "Below" on the last. Dragging a row over
 * the others renumbers the ports (one patch); a row's context menu is the keyboard path: Move up, Move down and Remove input. A Parallel lists
 * its inputs 1 to N without a handle or words, because their order changes nothing.
 */
function InputRows({ store, photo, rs, node, graph, stack }: InputRowsProps) {
  const settings = rs.settings;
  const states = cardStates(settings);
  const pictures = useAppState((s) => s.nodeThumbnails[photo.id]);
  const rows = inputsOf(settings, node);
  const shown = stack ? [...rows].reverse() : rows;
  const list = useRef<HTMLUListElement>(null);
  const [drag, setDrag] = useState<{
    from: number;
    over: number;
    dy: number;
  } | null>(null);
  void store;

  /** The place a pointer is over, counted in the shown order. */
  const indexAt = (clientY: number): number => {
    const els = [...(list.current?.querySelectorAll<HTMLElement>("[data-input-row]") ?? [])];
    for (let i = 0; i < els.length; i++) {
      const r = (els[i] as HTMLElement).getBoundingClientRect();
      if (clientY < r.top + r.height / 2) return i;
    }
    return Math.max(0, els.length - 1);
  };
  const onGripDown = (event: ReactPointerEvent<HTMLElement>, index: number): void => {
    if (event.button !== 0 || rows.length < 2) return;
    event.preventDefault();
    const startY = event.clientY;
    let latest = { from: index, over: index, dy: 0 };
    setDrag(latest);
    const onMove = (e: PointerEvent): void => {
      latest = {
        from: index,
        over: indexAt(e.clientY),
        dy: e.clientY - startY,
      };
      setDrag(latest);
    };
    const finish = (commit: boolean): void => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onCancel);
      setDrag(null);
      // The rows are shown top first: place `i` of the list is input `N - i`.
      if (commit && latest.over !== latest.from) graph.moveInput(node, rows.length - latest.from, rows.length - latest.over);
    };
    const onUp = (): void => finish(true);
    const onCancel = (): void => finish(false);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onCancel);
  };

  return (
    <ul ref={list} className="ym-input-rows" aria-label="Inputs" data-stack={stack || undefined}>
      {shown.map((row: InputRow, i) => {
        const number = inputNumber(row.port) ?? 1;
        const from: NodeRef = row.link?.from ?? {
          node: "input",
          port: "picture",
        };
        const name = feederName(settings, from, states);
        const last = shown.length - 1;
        const word = stack && rows.length > 1 ? (i === 0 ? "On top" : i === last ? "Below" : null) : null;
        const dragging = drag?.from === i;
        const over = drag !== null && drag.over === i && drag.from !== i;
        const onKeyDown = (event: KeyboardEvent<HTMLLIElement>): void => {
          if (!stack || !row.link || !event.altKey) return;
          // The keyboard path for the order is the row's menu; Alt with the arrows is the shortcut for it.
          if (event.key === "ArrowUp" && number < rows.length) graph.moveInput(node, number, number + 1);
          else if (event.key === "ArrowDown" && number > 1) graph.moveInput(node, number, number - 1);
          else return;
          event.preventDefault();
        };
        return (
          <ContextMenu key={`${row.port}:${name}`}>
            <ContextMenuTrigger asChild>
              {/* glasscn's `Item`, as the Masks card's rows are. */}
              <Item asChild size="sm" className="ym-input-row">
                <li
                  tabIndex={0}
                  data-input-row={number}
                  data-dragging={dragging || undefined}
                  data-over={over || undefined}
                  aria-label={`Input ${number}, ${name}${word ? `, ${word.toLowerCase()}` : ""}`}
                  style={dragging ? { transform: `translateY(${drag.dy}px)` } : undefined}
                  onKeyDown={onKeyDown}
                >
                  {stack ? (
                    <span className="ym-input-row__grip" aria-hidden="true" data-grip onPointerDown={(e) => onGripDown(e, i)}>
                      <GripVerticalIcon />
                    </span>
                  ) : null}
                  <RowPicture settings={settings} from={from} pictures={pictures} />
                  <span className="ym-input-row__num">{number}</span>
                  <span className="ym-input-row__name">{name}</span>
                  {word ? <span className="ym-input-row__sub">{word}</span> : null}
                </li>
              </Item>
            </ContextMenuTrigger>
            <ContextMenuContent aria-label={`Input ${number} actions`}>
              {stack ? (
                <>
                  <ContextMenuItem disabled={number >= rows.length} onSelect={() => graph.moveInput(node, number, number + 1)}>
                    Move up
                  </ContextMenuItem>
                  <ContextMenuItem disabled={number <= 1} onSelect={() => graph.moveInput(node, number, number - 1)}>
                    Move down
                  </ContextMenuItem>
                </>
              ) : null}
              <ContextMenuItem variant="destructive" disabled={!row.link} onSelect={() => graph.removeInput(node, row.port)}>
                Remove input
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
        );
      })}
    </ul>
  );
}

/** "They start from node 1, Main." and its variants (`docs/25` §5.5). */
export function baseSentence(settings: DevelopSettings, node: number): string {
  const base = baseOf(settings, node);
  return `They start from ${describeEnd(settings, base)}.`;
}

/** Parallel › Inputs: what the node does, where the inputs start from, and the inputs 1 to N. */
export function ParallelInputsTab(props: MixerTabProps) {
  return (
    <Card title="Inputs" last>
      <RowNote className="ym-mixer-line">Adds the change each input makes to the picture they start from.</RowNote>
      <RowNote className="ym-mixer-line" role="note">
        {baseSentence(props.rs.settings, props.node)}
      </RowNote>
      <InputRows {...props} stack={false} />
    </Card>
  );
}

/** Layer › Blend: the Blend card (Mode, Opacity, Reset) and the Inputs card, from the top of the stack. */
export function LayerBlendTab(props: MixerTabProps) {
  const { store, rs, node } = props;
  const item = nodeOf(rs.settings, node);
  if (item?.kind !== "layer") return null;
  const send = (partial: { mode?: BlendMode; opacity?: number }, final = true): void => void store.update(patchOf(node, partial), final, rs.photo_id);
  return (
    <>
      <Card title="Blend" action={<ResetLink size="card" onClick={() => send({ mode: "normal", opacity: 1 })} />}>
        <Field label="Mode">
          <Select value={item.mode} onValueChange={(mode) => send({ mode: mode as BlendMode })}>
            <SelectTrigger aria-label="Blend mode" className="ym-film-select">
              <SelectValue>{BLEND_MODES.find((m) => m.value === item.mode)?.label ?? item.mode}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {BLEND_MODES.map((mode) => (
                <SelectItem key={mode.value} value={mode.value}>
                  {mode.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
        <Slider
          label="Opacity"
          value={Math.round(item.opacity * 100)}
          min={0}
          max={100}
          step={1}
          // No `defaultValue`: the fill runs from 0, as design/mixers draws it (a detent at 100 % would leave a full slider unfilled); Reset is the card's.
          format={formats.percent}
          onChange={(percent, final) => send({ opacity: percent / 100 }, final)}
        />
      </Card>
      <Card title="Inputs" last>
        <InputRows {...props} stack />
        <RowNote>Drag a row to change which input sits on top.</RowNote>
      </Card>
    </>
  );
}

/** A channel's chip: its letter in its tint. */
function ChannelChip({ port }: { port: string }) {
  return (
    <span className="ym-channel-row__chip" style={{ "--ch": CHANNEL_TINT[port] } as React.CSSProperties}>
      {CHANNEL_LETTER[port]}
    </span>
  );
}

const CHANNEL_NAMES: Readonly<Record<string, string>> = {
  red: "Red",
  green: "Green",
  blue: "Blue",
};

/** The Splitter's and the Combiner's Channels tab: the sentence of the node and a row for each channel with where it goes or comes from. */
export function ChannelsTab({ rs, node }: Pick<MixerTabProps, "rs" | "node">) {
  const settings = rs.settings;
  const item = nodeOf(settings, node);
  if (!isGlyphItem(item) || (item.kind !== "splitter" && item.kind !== "combiner")) return null;
  const states = cardStates(settings);
  const split = item.kind === "splitter";
  const where = (port: (typeof CHANNEL_PORTS)[number]): ReactNode => {
    if (split) {
      const to = settings.nodes.links.filter((l: NodeLink) => l.from.node === node && l.from.port === port).map((l) => describeEnd(settings, l.to, states));
      return to.length > 0 ? `To ${joinWords(to)}` : "Not linked";
    }
    const from = settings.nodes.links.find((l) => l.to.node === node && l.to.port === port)?.from;
    return from ? `From ${describeEnd(settings, from, states)}` : "From the input";
  };
  return (
    <Card title="Channels" last>
      <RowNote className="ym-mixer-line">
        {split
          ? "Sends the red, green and blue of the picture to three outputs. Each output is a grey picture of one channel, so the nodes after it grade that channel."
          : "Makes one picture: the red of its red input, the green of its green input and the blue of its blue input."}
      </RowNote>
      <ul className="ym-channel-rows" aria-label="Channels">
        {CHANNEL_PORTS.map((port) => (
          <Item key={port} asChild size="sm" className="ym-channel-row">
            <li data-channel={port}>
              <ChannelChip port={port} />
              <span className="ym-channel-row__name">{CHANNEL_NAMES[port]}</span>
              <span className="ym-channel-row__where">{where(port)}</span>
            </li>
          </Item>
        ))}
      </ul>
    </Card>
  );
}
