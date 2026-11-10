import { BanIcon, LockIcon, RotateCcwIcon } from "lucide-react";
import { useEffect, useRef } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { Input } from "@/components/glass/input";
import type { DevelopSettings, NodeId } from "../../api/types";
import { Button, panelKindOf } from "../../components";
import { cardStates, chipText, GRAPH_OP_TEXT, headLine } from "../../state/graphOps";
import { isGlyphItem, nodeOf } from "../../state/settingsGraph";
import { KindGlyph } from "./GlyphCard";

/** The tooltip and accessible name of the head's Reset, by kind of node (`docs/23` §7.5). */
const RESET_TEXT = { input: "Reset input", output: "Reset output", corrector: "Reset this node", film: "Reset this node", parallel: "Reset this node", layer: "Reset this node", splitter: "Reset this node", combiner: "Reset this node" } as const;

export interface NodeHeadProps {
  settings: DevelopSettings;
  /** The selected node. */
  node: NodeId;
  /** The name is a text box (a double click, or More › Rename), and what ends it: the name typed, or null for Esc. */
  renaming: boolean;
  onStartRename: () => void;
  onRename: (name: string | null) => void;
  /** Bypass this node (B): a toggle, for a corrector. */
  onBypass: () => void;
  /** Reset this node: one undo step, its masks, name and links kept. */
  onReset: () => void;
  /** The More menu of this node, to sit after the icons (`NodeMoreMenu`); the Output has none. */
  more?: ReactNode;
}

/**
 * The node's head, the top of the panel in the node view (`docs/23` §7.5, `design/nodes/05-nodes-default.jpg`): one line of 52 px with the
 * **number chip** (a lock for the Input and the Output), the **name** at 20 px bold (a double click renames a corrector), a quiet line
 * that says what the node is ("Corrector · whole photo", "Corrector · 2 masks", "Turns the camera's data into the scene.", "Cine rendering,
 * look and export"), and at the right round icon buttons: **Bypass** (a corrector; a toggle, "Bypass this node (B)"), **Reset** ("Reset this
 * node", "Reset input", "Reset output") and **More**. The 40 px title block of the simple view is not here.
 */
export function NodeHead({ settings, node, renaming, onStartRename, onRename, onBypass, onReset, more }: NodeHeadProps) {
  const item = nodeOf(settings, node);
  const field = useRef<HTMLInputElement>(null);
  const finished = useRef(false);
  const name = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (!renaming) return;
    finished.current = false;
    field.current?.focus();
    field.current?.select();
  }, [renaming]);
  if (!item) return null;
  const states = cardStates(settings);
  const glyph = isGlyphItem(item);
  // Every node but the Input and the Output has a name, a number or a glyph in the chip, and is renamed and bypassed (a Splitter and a Combiner cannot be).
  const corrector = item.kind === "corrector" || item.kind === "film" || glyph;
  const noBypass = item.kind === "splitter" || item.kind === "combiner";
  // A Parallel, a Splitter and a Combiner have nothing to reset (`docs/25` §5.5).
  const nothingToReset = item.kind === "parallel" || item.kind === "splitter" || item.kind === "combiner";
  const title = item.kind === "input" ? "Input" : item.kind === "output" ? "Output" : item.name;
  const finish = (value: string | null, restoreFocus: boolean): void => {
    if (finished.current) return;
    finished.current = true;
    onRename(value);
    // A key ended it: the focus goes to the name, which is where it was; leaving the box by a click leaves it where the person put it.
    if (restoreFocus) requestAnimationFrame(() => name.current?.focus());
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
    <div className="ym-nodehead" role="group" aria-label="Node" data-kind={item.kind}>
      <span className="ym-nodehead__chip" data-lock={!corrector || undefined} data-glyph={glyph || undefined}>
        {glyph ? <KindGlyph kind={item.kind} /> : corrector ? chipText(states.get(item.id as number)?.number ?? 0) : <LockIcon aria-hidden="true" />}
      </span>
      <div className="ym-nodehead__text">
        {renaming && corrector ? (
          <Input
            ref={field}
            size="sm"
            className="ym-nodehead__rename"
            aria-label="Node name"
            maxLength={32}
            defaultValue={item.name}
            spellCheck={false}
            onKeyDown={onKeyDown}
            onBlur={(event) => finish(event.currentTarget.value, false)}
          />
        ) : (
          <h2
            ref={name}
            className="ym-nodehead__name"
            tabIndex={corrector ? -1 : undefined}
            title={corrector ? "Double-click to rename" : undefined}
            onDoubleClick={corrector ? onStartRename : undefined}
          >
            {title}
          </h2>
        )}
        <p className="ym-nodehead__line">{headLine(settings, node, states)}</p>
      </div>
      <div className="ym-nodehead__actions" role="group" aria-label="Node actions">
        {corrector ? (
          <Button
            variant="header"
            pressed={"bypass" in item && item.bypass}
            disabled={noBypass}
            className="ym-nodehead__btn"
            aria-label="Bypass this node"
            title={noBypass ? GRAPH_OP_TEXT.glyphNoBypass : "Bypass this node (B)"}
            icon={<BanIcon aria-hidden="true" />}
            onClick={onBypass}
          />
        ) : null}
        <Button
          variant="header"
          className="ym-nodehead__btn"
          disabled={nothingToReset}
          aria-label={RESET_TEXT[panelKindOf(item.kind)]}
          title={nothingToReset ? GRAPH_OP_TEXT.nothingToReset : RESET_TEXT[panelKindOf(item.kind)]}
          icon={<RotateCcwIcon aria-hidden="true" />} onClick={onReset} />
        {more}
      </div>
    </div>
  );
}
