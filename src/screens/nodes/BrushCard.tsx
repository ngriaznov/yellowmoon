import type { MouseEvent } from "react";
import type { MaskItem, MaskPatch, ResolvedSettings } from "../../api/types";
import { Card, formats, ResetLink, RowNote, Segmented, Slider } from "../../components";
import type { SegmentedOption } from "../../components";
import { useAppState } from "../../state/context";
import { DEFAULT_BRUSH } from "../../state/brushStrokes";
import { newMask, resetRowPatch } from "../../state/maskRows";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { sizeField, sizeSlider, unitField, unitSlider } from "../tools/maskScales";
import { RasterSliders } from "./MaskSliders";

type BrushRow = Extract<MaskItem, { kind: "brush" }>;
type Tool = "paint" | "erase";

/** The card's tool: Paint adds to the mask, Erase takes from it. */
const TOOLS: ReadonlyArray<SegmentedOption<Tool>> = [
  { value: "paint", label: "Paint" },
  { value: "erase", label: "Erase" },
];

/** What the line under the tool says: how to draw, or how to get the tool back after Esc. */
function toolLine(on: boolean, erase: boolean, enabled: boolean): string {
  if (!enabled) return "This mask is off. Turn it on to draw.";
  if (!on) return "Choose Paint or Erase to draw on the photo.";
  return erase ? "Drag on the photo to take paint away. Esc stops." : "Drag on the photo to paint. Esc stops.";
}

export interface BrushCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  /** The corrector the row is in, the row's index in its `masks.items`, and the row. */
  node: number;
  index: number;
  row: BrushRow;
}

/**
 * The card of a selected Brush row (`docs/23` §7.6): **Paint · Erase**, **Size** and **Hardness**, then the raster sliders (**Refine edges**
 * at 0 for a Brush, **Feather**, **Grow**, **Opacity**). Paint · Erase, Size and Hardness are the brush the **next** strokes are drawn with
 * (`store.brush`: the strokes already drawn keep theirs, and nothing of it is a patch or an undo step); Size is `0.001 + 0.249 (x / 100)²` of the
 * long edge (`maskScales.ts`), so a small brush is within reach. Choosing Paint or Erase, the chosen one too, turns the Brush tool on, so that
 * the way back from Esc is on the card. The four raster sliders each write one field of the row as a `masks.edit` row patch, and Reset puts the
 * row back as a new Brush row in the same mode (no strokes, every field at its default) and the brush at its default.
 */
export function BrushCard({ store, rs, node, index, row }: BrushCardProps) {
  const brush = useAppState((s) => s.brush);
  const tool = useAppState((s) => s.brushTool);
  const send = (edit: MaskPatch, final: boolean): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: edit } } }), final, rs.photo_id);
  const defaults = newMask("brush", row.mode) as BrushRow;
  const reset = (): void => {
    store.setBrush(DEFAULT_BRUSH);
    void store.update(patchOf(node, { masks: { edit: { [String(index)]: resetRowPatch(row) } } }), true, rs.photo_id);
  };
  const choose = (next: Tool): void => {
    store.setBrush({ erase: next === "erase" });
    store.setBrushTool(true);
  };
  // A click on the chosen segment is not a change for the control, but it is a request for the tool.
  const onClick = (event: MouseEvent<HTMLDivElement>): void => {
    const segment = (event.target as HTMLElement).closest<HTMLElement>('[role="radio"]');
    const value = segment?.dataset.value;
    if (value === "paint" || value === "erase") choose(value);
  };
  return (
    <Card title="Brush" action={<ResetLink size="card" onClick={reset} />}>
      <div className="ym-brush-tool" onClick={onClick}>
        <Segmented label="Paint or erase" value={brush.erase ? "erase" : "paint"} onChange={choose} options={TOOLS} />
      </div>
      <RowNote>{toolLine(tool, brush.erase, row.enabled)}</RowNote>
      <Slider
        label="Size"
        value={sizeSlider(brush.size)}
        min={0}
        max={100}
        defaultValue={sizeSlider(DEFAULT_BRUSH.size)}
        format={formats.integer}
        onChange={(v) => store.setBrush({ size: sizeField(v) })}
        onReset={() => store.setBrush({ size: DEFAULT_BRUSH.size })}
      />
      <Slider
        label="Hardness"
        value={unitSlider(brush.hardness)}
        min={0}
        max={100}
        defaultValue={unitSlider(DEFAULT_BRUSH.hardness)}
        format={formats.integer}
        onChange={(v) => store.setBrush({ hardness: unitField(v) })}
        onReset={() => store.setBrush({ hardness: DEFAULT_BRUSH.hardness })}
      />
      <RasterSliders row={row} defaults={defaults} send={send} />
    </Card>
  );
}
