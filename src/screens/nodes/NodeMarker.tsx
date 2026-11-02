import type { NodeProps } from "@xyflow/react";
import { LockIcon } from "lucide-react";
import type { KeyboardEvent } from "react";
import type { NodeId } from "../../api/types";
import { Port } from "./Port";
import { useGraphUi } from "./graphUi";

/**
 * The Input and the Output (`docs/23` §7.2): fixed round markers of 32 px with a lock glyph and the name under them; a square output port
 * at the Input's right and an arrowhead input port at the Output's left. They are selectable like a card (the panel then edits them), they
 * cannot be moved, renamed, bypassed or deleted, and their names are "Input" and "Output".
 */
export function NodeMarker({ data }: NodeProps) {
  const ui = useGraphUi();
  const id = (data as { id: NodeId }).id;
  const which = id === "input" ? "input" : "output";
  const word = which === "input" ? "Input" : "Output";
  const selected = ui.selected === id;
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
    <div className="ym-marker" data-which={which} data-selected={selected || undefined} data-quiet={(selected && ui.linkSelected) || undefined}>
      <div
        className="ym-marker__dot"
        role="button"
        tabIndex={selected ? 0 : -1}
        aria-pressed={selected}
        aria-label={word}
        data-node-card={String(id)}
        onClick={() => ui.onSelect(id)}
        onKeyDown={onKeyDown}
      >
        <LockIcon aria-hidden="true" />
      </div>
      <span className="ym-marker__word" aria-hidden="true">
        {word}
      </span>
      <Port node={id} port="picture" side={which === "input" ? "out" : "in"} on={which} />
    </div>
  );
}
