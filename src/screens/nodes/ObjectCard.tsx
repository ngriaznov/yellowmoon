import type { MouseEvent } from "react";
import type { MaskPatch, ResolvedSettings } from "../../api/types";
import { Card, ResetLink, RowNote, Segmented } from "../../components";
import type { SegmentedOption } from "../../components";
import { useAppState } from "../../state/context";
import { newMask, resetRowPatch } from "../../state/maskRows";
import { DEFAULT_OBJECT_MODE, objectLine } from "../../state/objectMask";
import type { ObjectMode, ObjectRow } from "../../state/objectMask";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { RasterSliders } from "./MaskSliders";

/** The card's tool: what a plain click on the photo does (`docs/23` §7.6). */
const TOOLS: ReadonlyArray<SegmentedOption<ObjectMode>> = [
  { value: "add", label: "Add point" },
  { value: "remove", label: "Remove point" },
  { value: "box", label: "Box" },
];

export interface ObjectCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  /** The corrector the row is in, the row's index in its `masks.items`, and the row. */
  node: number;
  index: number;
  row: ObjectRow;
}

/**
 * The card of a selected Object row (`docs/23` §7.6): the tool **Add point · Remove point · Box** with its line ("Click the photo to add to the object.
 * ⌥-click to take a part away."), then the raster sliders (**Refine edges** at 50 %, **Feather**, **Grow**, **Opacity**). The tool is what a plain click
 * on the photo does (`store.objectMode`: a window setting, no patch and no undo step); choosing a tool, the chosen one too, turns the Object tool on,
 * so that the way back from Esc is on the card. The four sliders each write one field of the row as a `masks.edit` row patch, and Reset puts the row back as
 * a new Object row in the same mode (no point, no box, every field at its default) and the tool at Add point.
 */
export function ObjectCard({ store, rs, node, index, row }: ObjectCardProps) {
  const mode = useAppState((s) => s.objectMode);
  const on = useAppState((s) => s.objectTool);
  const send = (edit: MaskPatch, final: boolean): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: edit } } }), final, rs.photo_id);
  const defaults = newMask("object", row.mode) as ObjectRow;
  const reset = (): void => {
    store.setObjectMode(DEFAULT_OBJECT_MODE);
    void store.update(patchOf(node, { masks: { edit: { [String(index)]: resetRowPatch(row) } } }), true, rs.photo_id);
  };
  const choose = (next: ObjectMode): void => {
    store.setObjectMode(next);
    store.setObjectTool(true);
  };
  // A click on the chosen segment is not a change for the control, but it is a request for the tool.
  const onClick = (event: MouseEvent<HTMLDivElement>): void => {
    const segment = (event.target as HTMLElement).closest<HTMLElement>('[role="radio"]');
    const value = segment?.dataset.value;
    if (value === "add" || value === "remove" || value === "box") choose(value);
  };
  return (
    <Card title="Object" action={<ResetLink size="card" onClick={reset} />}>
      <div className="ym-object-tool" onClick={onClick}>
        <Segmented label="Object tool" value={mode} onChange={choose} options={TOOLS} />
      </div>
      <RowNote>{objectLine(mode, on, row.enabled)}</RowNote>
      <RasterSliders row={row} defaults={defaults} send={send} />
    </Card>
  );
}
