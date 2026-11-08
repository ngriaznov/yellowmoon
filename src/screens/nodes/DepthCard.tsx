import { EyeIcon, FocusIcon, PipetteIcon } from "lucide-react";
import { useEffect } from "react";
import type { MaskItem, MaskPatch, ResolvedSettings } from "../../api/types";
import { Button, Card, ResetLink, RowNote } from "../../components";
import { useAppState } from "../../state/context";
import { depthPercent, isWholeDepth, newMask, resetRowPatch } from "../../state/maskRows";
import { patchOf } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { RasterSliders } from "./MaskSliders";

/** A Depth row's fields (`docs/23` §3.3). */
export type DepthRow = Extract<MaskItem, { kind: "depth" }>;

/** What the card says under Pick depth and Focus plane, word for word (`docs/23` §4.8, §7.6). */
export const NO_FOCUS_POINT = "This photo has no focus point.";
export const FOCUS_AT_CENTRE = "The camera may not have recorded where it focused. Check the result.";
/** What Focus plane and Show depth say as tooltips. */
export const FOCUS_PLANE_TIP = "Select what the camera focused on";
export const SHOW_DEPTH_TIP = "Show how near or far each part is. White is near.";
export const PICK_DEPTH_TIP = "Click the part of the photo to select";
/** What the card's read-only line says about where the mask ends (`docs/11`, the Depth card): at the whole depth, or at a share of it. */
export const FADES_AT_FARTHEST = "Fades out at the farthest point";
export const fadesOutAt = (far: number): string => (isWholeDepth(far) ? FADES_AT_FARTHEST : `Fades out at ${depthPercent(far)} % of the depth`);
/** The end of the card's sentence when the map could not be made; the core's own sentence ("The Depth mask can't be made: …") comes before it. */
export const FAILED_ADVICE = "Try again, or remove the mask.";

export interface DepthCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  /** The corrector the row is in, the row's index in its `masks.items`, and the row. */
  node: number;
  index: number;
  row: DepthRow;
  /** Pick depth is on (`Photos` keeps it with the other picks: one at a time, ended by the tab, the node, the row, Crop and the photo) and turns it on or off. */
  picking: boolean;
  onPick: (on: boolean) => void;
  /** The core's sentence when the photo's depth map failed ("The Depth mask can't be made: …"), else null. */
  failure: string | null;
}

/**
 * The card of a selected Depth row (`docs/23` §4.8, §7.6; `design/depth/` is the history of the band): **DEPTH** with its Reset; **Pick depth** (a toggle with an
 * eyedropper; on when the row is added) and **Focus plane** (a button, off with "This photo has no focus point." for a file with none, and with the line "The
 * camera may not have recorded where it focused. Check the result." while the limit is the one a focus point at the exact centre of the frame gave); a read-only
 * line that says where the mask ends ("Fades out at 35 % of the depth", "Fades out at the farthest point"); **Show depth**, a view toggle (no patch, no undo
 * step); and the raster sliders every raster card has. The mask is a gradient, white at the nearest part and none at the row's far limit, which only Pick depth
 * and Focus plane set (there is no slider for it). Each raster slider writes one field of the row as a `masks.edit` row patch (one undo step for a drag). When the
 * photo's depth map failed the card is the sentence and **Try again**, which starts the map again; nothing else is offered.
 */
export function DepthCard({ store, rs, node, index, row, picking, onPick, failure }: DepthCardProps) {
  const photo = rs.photo_id;
  const noFocus = useAppState((s) => s.noFocusPoint[photo] === true);
  const centre = useAppState((s) => s.focusCentre[photo]);
  const showing = useAppState((s) => s.showDepth === index);
  useEffect(() => store.ensureFocusPoint(photo), [store, photo]);
  // Show depth is the card's: it ends when the card goes (another row, another tab, another node).
  useEffect(
    () => () => {
      if (store.getState().showDepth === index) store.setShowDepth(null);
    },
    [store, index],
  );
  const send = (edit: MaskPatch, final: boolean): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: edit } } }), final, photo);
  const defaults = newMask("depth", row.mode) as DepthRow;
  const reset = (): void => void store.update(patchOf(node, { masks: { edit: { [String(index)]: resetRowPatch(row) } } }), true, photo);
  const action = <ResetLink size="card" onClick={reset} />;

  if (failure !== null) {
    return (
      <Card title="Depth" action={action}>
        <RowNote role="status" className="ym-depth-error">
          {failure} {FAILED_ADVICE}
        </RowNote>
        <div className="ym-mask-pick">
          <Button variant="header" size="sm" onClick={() => void store.retryDepth(photo)}>
            Try again
          </Button>
        </div>
      </Card>
    );
  }

  const checkCentre = centre !== undefined && centre === row.far;
  return (
    <Card title="Depth" action={action}>
      <div className="ym-mask-pick">
        <Button variant="header" size="sm" pressed={picking} icon={<PipetteIcon aria-hidden="true" />} title={PICK_DEPTH_TIP} onClick={() => onPick(!picking)}>
          Pick depth
        </Button>
        <span className="ym-mask-pick__right">
          <Button
            variant="header"
            size="sm"
            icon={<FocusIcon aria-hidden="true" />}
            disabled={noFocus}
            title={noFocus ? NO_FOCUS_POINT : FOCUS_PLANE_TIP}
            onClick={() => void store.focusPlane(node, index)}
          >
            Focus plane
          </Button>
        </span>
      </div>
      {noFocus ? <RowNote>{NO_FOCUS_POINT}</RowNote> : checkCentre ? <RowNote role="status">{FOCUS_AT_CENTRE}</RowNote> : null}
      <div className="ym-mask-pick">
        <Button variant="header" size="sm" pressed={showing} icon={<EyeIcon aria-hidden="true" />} title={SHOW_DEPTH_TIP} onClick={() => store.setShowDepth(showing ? null : index)}>
          Show depth
        </Button>
      </div>
      <RowNote className="ym-depth-fade">{fadesOutAt(row.far)}</RowNote>
      <RasterSliders row={row} defaults={defaults} send={send} />
    </Card>
  );
}
