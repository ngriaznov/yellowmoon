import { useEffect, useMemo } from "react";
import type { KeyboardEvent, MouseEvent } from "react";
import type { MaskKind, MaskMode, MasksPatch, MaskStyle, ResolvedSettings } from "../../api/types";
import { Button, Card, Field, KeyHint, ResetLink, RowNote, Segmented } from "../../components";
import type { SegmentedOption } from "../../components";
import { useAppState, useStore } from "../../state/context";
import { editRow, isUnsupported, MASKS_FULL, MAX_MASKS, movedMask, newMask, selectionAfterRemove, withMask, withoutMask } from "../../state/maskRows";
import type { VisionRows } from "../../state/maskRows";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { AddMaskMenu } from "./AddMaskMenu";
import { CARD_KINDS, MaskCard } from "./MaskCards";
import { InvertGlyph, MaskRows } from "./MaskRows";

/** The kinds Apple's Vision makes (`docs/23` §3.10): the Add mask menu lists them only where the core says this computer can make them. */
const VISION_KINDS: readonly MaskKind[] = ["subject", "people"];

/** The Depth card's Pick depth where the screen has none. */
const NO_PICK = (): void => {};

/** What the Masks card says under its rows (`docs/23` §7.6), word for word. */
export const MASKS_ADD_UP = "Masks add up. Subtract cuts a mask out. Colour and brightness ranges keep only their part of the rest.";

/** The View card's two segments (`docs/23` §7.6). The mockups draw a third, "Off"; the toolbar's Show mask and M turn the view on and off. */
export const MASK_VIEW_OPTIONS: ReadonlyArray<SegmentedOption<MaskStyle>> = [
  { value: "overlay", label: "Overlay" },
  { value: "mask_only", label: "Mask only" },
];

/**
 * The View card (`docs/23` §7.6): "VIEW" with the key chip M, and one row, Show mask, with **Overlay · Mask only**. Choosing a segment sets the
 * style and turns the view on, also when it is the segment that is already chosen (the view may be off). The style is kept with the view.
 */
export function ViewCard() {
  const store = useStore();
  const { style } = useAppState((s) => s.maskView);
  // A click on the chosen segment is not a change for the control, but it is a request for the view.
  const onClick = (event: MouseEvent<HTMLDivElement>): void => {
    const segment = (event.target as HTMLElement).closest<HTMLElement>('[role="radio"]');
    const value = segment?.dataset.value;
    if (value === "overlay" || value === "mask_only") store.setMaskView({ style: value });
  };
  return (
    <Card
      title="View"
      last
      action={
        <span className="ym-card__keys" aria-hidden="true">
          <KeyHint>M</KeyHint>
        </span>
      }
    >
      <Field label="Show mask">
        <div className="ym-view-style" onClick={onClick}>
          <Segmented label="Show mask" value={style} onChange={(next) => store.setMaskView({ style: next })} options={MASK_VIEW_OPTIONS} />
        </div>
      </Field>
    </Card>
  );
}

export interface MaskTabProps {
  store: AppStore;
  rs: ResolvedSettings;
  /** The corrector the panel is on. */
  node: number;
  /** Pick colour of a Colour range is on, and turns it on or off (`Photos` keeps it with the other picks: one at a time, off when the tab, the node, Crop or the photo changes). */
  pickingRange: boolean;
  onPickRange: (on: boolean) => void;
  /** Pick depth of a Depth row is on, and turns it on or off (`Photos` keeps it with the other picks). Left out, there is no such pick (a test's screen). */
  pickingDepth?: boolean;
  onPickDepth?: (on: boolean) => void;
}

/**
 * A corrector's Mask tab (`docs/23` §7.6): the **Masks card** (its header "MASKS", **Add mask ▾**, **Invert**, which is the node's whole
 * combined mask, and **Reset**, which removes every row and un-inverts; then the rows and the line "Masks add up…"), the **selected mask's
 * card** (an Object, a Background, a Linear, a Radial, a Brush, a Colour range or a Brightness range: Subject and People have theirs with their task, and a raster row that a
 * file or a paste brought shows its row and no card), and the View card. Adding a Brush row, and choosing one, starts the Brush tool. Every change is one node patch: adding, removing and moving a row
 * is `masks.items`, a row's mode, Invert and eye and every slider is a `masks.edit` row patch, the node's Invert is `masks.invert`; each
 * is one undo step (a slider's drag is one).
 */
export function MaskTab({ store, rs, node, pickingRange, onPickRange, pickingDepth = false, onPickDepth }: MaskTabProps) {
  const item = nodeOf(rs.settings, node);
  const corrector = item?.kind === "corrector" || item?.kind === "film" ? item : null;
  const rows = corrector?.masks.items ?? [];
  const inverted = corrector?.masks.invert ?? false;
  const stored = useAppState((s) => s.maskRow);
  const kinds = useAppState((s) => s.maskKinds);
  const eye = useAppState((s) => s.maskEye);
  const finding = useAppState((s) => s.viewportBusy.busy && s.viewportBusy.finding);
  // SAM's encoder is getting the photo ready, or a click is being answered (`docs/23` §7.6): the Object rows say so.
  const preparing = useAppState((s) => s.objectPreparing.includes(rs.photo_id));
  const objectFinding = useAppState((s) => s.objectFinding === rs.photo_id);
  const objectBusy = useMemo(() => ({ preparing, finding: objectFinding }), [preparing, objectFinding]);
  // Subject and People (`docs/23` §3.10, §7.6): a row of a kind this computer cannot make (the kinds `mask_kinds` leaves out) says so and selects nothing;
  // one whose mask found nothing says "Nothing found". The kinds are not known at first: no row says it cannot until they are.
  const nothing = useAppState((s) => s.visionNothing[rs.photo_id]);
  // Depth (`docs/23` §3.10a, §4.8): a row where the weights are missing says so and has no card; one whose map failed says "Can't be made".
  const depthFailed = useAppState((s) => s.depthFailed[rs.photo_id] ?? null);
  const vision = useMemo<VisionRows>(
    () => ({
      unsupported: kinds === null ? [] : VISION_KINDS.filter((k) => !kinds.includes(k)),
      nothing: nothing ?? [],
      depthMissing: kinds !== null && !kinds.includes("depth"),
      depthFailed: depthFailed !== null,
    }),
    [kinds, nothing, depthFailed],
  );
  const selected = stored !== null && stored < rows.length ? stored : null;
  const row = selected === null ? undefined : rows[selected];
  useEffect(() => store.ensureMaskKinds(), [store]);

  /** Sends a node patch of this node's masks as one undo step. */
  const send = (masks: MasksPatch): void => void store.update(patchOf(node, { masks }), true, rs.photo_id);

  const add = (kind: MaskKind): void => {
    const patch = withMask(rows, newMask(kind));
    if (!patch) return;
    store.selectMaskRow(rows.length);
    send(patch);
    // A Colour range's tool is Pick colour: the next click on the photo is its colour. A Brush's is the brush: the next drag on the photo paints. An
    // Object's is the click: the next click on the photo finds the object.
    onPickRange(kind === "color_range");
    // A Depth row's is Pick depth: the next click on the photo picks the depth (`docs/23` §4.8).
    onPickDepth?.(kind === "depth");
    store.setBrushTool(kind === "brush");
    store.setObjectTool(kind === "object");
  };
  /** Choosing a row selects it, and a Brush or an Object row starts its tool (Esc ended the last one, and this is the way back). */
  const select = (index: number): void => {
    store.selectMaskRow(index);
    if (rows[index]?.kind === "brush") store.setBrushTool(true);
    if (rows[index]?.kind === "object") store.setObjectTool(true);
  };
  const remove = (index: number): void => {
    store.setMaskEye(null);
    send(withoutMask(rows, index));
    store.selectMaskRow(selectionAfterRemove(index, rows.length));
    onPickRange(false);
    onPickDepth?.(false);
  };
  const move = (index: number, by: -1 | 1): void => {
    const patch = movedMask(rows, index, by);
    if (!patch) return;
    store.setMaskEye(null);
    send(patch);
    if (selected === index) store.selectMaskRow(index + by);
    else if (selected === index + by) store.selectMaskRow(index);
  };
  const edit = (index: number, change: { mode?: MaskMode; invert?: boolean; enabled?: boolean }): void => send(editRow(index, change));
  const reset = (): void => {
    store.setMaskEye(null);
    send({ items: [], invert: false });
    store.selectMaskRow(null);
    onPickRange(false);
    onPickDepth?.(false);
  };

  /**
   * Delete and Backspace in the Masks card (`docs/23` §7.7): they take out the row that has the focus, else the selected row, and never the node
   * (the graph's Delete leaves the card alone: `data-mask-card`). Not from a menu: a menu's key events reach here through React's tree, not the DOM.
   */
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    // Return on the selected Object row starts "Place points" (`docs/28` A-11): the keyboard's way to add the points a click adds.
    if (event.key === "Enter" && !event.shiftKey && !event.altKey && !event.metaKey && !event.ctrlKey && !event.repeat) {
      const button = (event.target as HTMLElement).closest<HTMLElement>("button.ym-mask-row__select");
      const index = Number(button?.closest<HTMLElement>("li[data-index]")?.dataset.index);
      const target = rows[index];
      if (button && index === selected && target?.kind === "object" && target.enabled && event.currentTarget.contains(button)) {
        event.preventDefault();
        store.setObjectPlacing(true);
      }
      return;
    }
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    if (event.shiftKey || event.altKey || event.metaKey || event.ctrlKey || event.repeat) return;
    if (!event.currentTarget.contains(event.target as Node)) return;
    event.preventDefault();
    const inRow = (event.target as HTMLElement).closest<HTMLElement>("li[data-index]");
    const index = inRow ? Number(inRow.dataset.index) : selected;
    if (index !== null && index < rows.length) remove(index);
  };

  return (
    <>
      <div className="ym-mask-card" data-mask-card="" onKeyDown={onKeyDown}>
        <Card
          title="Masks"
          action={
            <>
              <AddMaskMenu kinds={kinds} onAdd={add} blocked={rows.length >= MAX_MASKS ? MASKS_FULL : null} />
              <Button variant="header" size="sm" pressed={inverted} icon={<InvertGlyph />} title="Invert the whole mask of this node" onClick={() => send({ invert: !inverted })}>
                Invert
              </Button>
              {rows.length > 0 || inverted ? <ResetLink size="card" onClick={reset} /> : null}
            </>
          }
        >
          <MaskRows rows={rows} selected={selected} eye={eye} onEye={(index) => store.setMaskEye(index)} onSelect={select} onEdit={edit} onRemove={remove} onMove={move} finding={finding} object={objectBusy} vision={vision} />
          <RowNote>{MASKS_ADD_UP}</RowNote>
        </Card>
      </div>
      {row && selected !== null && CARD_KINDS.has(row.kind) && !isUnsupported(row, vision) ? (
        <MaskCard
          key={`${selected}-${row.kind}`}
          store={store}
          rs={rs}
          node={node}
          index={selected}
          row={row}
          pickingRange={pickingRange && row.kind === "color_range"}
          onPickRange={onPickRange}
          pickingDepth={pickingDepth && row.kind === "depth"}
          onPickDepth={onPickDepth ?? NO_PICK}
          failure={row.kind === "depth" ? depthFailed : null}
        />
      ) : null}
      <ViewCard />
    </>
  );
}
