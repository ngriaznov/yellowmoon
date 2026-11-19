import { Card, ResetLink, Segmented, Switch } from "../../components";
import type { SegmentedOption } from "../../components";
import type { BlurNode, BlurType, ResolvedSettings } from "../../api/types";
import { nodeOf, patchOf } from "../../state/settingsGraph";
import type { BlurItem } from "../../state/settingsGraph";
import { newBlur } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";
import { BLUR_TEXT, BLUR_TYPE_READY } from "./blurParts";
import type { BlurControlProps } from "./blurParts";
import { LensControls } from "./LensControls";
import { MotionControls } from "./MotionControls";
import { SoftControls } from "./SoftControls";
import { ZoomControls } from "./ZoomControls";

const TYPES: ReadonlyArray<[BlurType, string]> = [
  ["soft", "Soft"],
  ["lens", "Lens"],
  ["motion", "Motion"],
  ["zoom", "Zoom"],
];

/** Whether anything differs from a new node's blur settings (the masks, name and bypass do not count). */
function changed(item: BlurNode): boolean {
  const d = newBlur(0, [0, 0]);
  const same = (a: unknown, b: unknown): boolean => JSON.stringify(a) === JSON.stringify(b);
  return item.type !== d.type || !same(item.soft, d.soft) || !same(item.lens, d.lens) || !same(item.motion, d.motion) || !same(item.zoom, d.zoom) || item.respect_mask_edges !== d.respect_mask_edges;
}

export interface BlurTabProps {
  store: AppStore;
  rs: ResolvedSettings;
  node: number;
  item: BlurItem;
  /** Pick centre of the Zoom type is on, and turns it on or off (`Photos` keeps it, as the other picks). */
  pickingCentre?: boolean;
  onPickCentre?: (on: boolean) => void;
}

/**
 * A Blur node's Blur tab (`docs/33` §2.1, §2.2, §2.3, §2.6): the Blur card (the Type control, then the chosen type's controls) and the Mask
 * edges card (the switch and its line). Every control writes the node as one patch; a drag is a patch per frame that folds into one undo step,
 * and the card's Reset is one patch of the type and every value (the masks stay). Switching type changes `type` alone, so each type keeps its values.
 */
export function BlurTab({ store, rs, node, item, pickingCentre = false, onPickCentre }: BlurTabProps) {
  const send: BlurControlProps["send"] = (partial, final = true) => void store.update(patchOf(node, partial), final, rs.photo_id);
  const options: ReadonlyArray<SegmentedOption<BlurType>> = TYPES.map(([value, label]) => ({ value, label, disabled: !BLUR_TYPE_READY[value] && value !== item.type }));
  const defaults = newBlur(0, [0, 0]);
  const reset = (): void => send({ type: defaults.type, soft: defaults.soft, lens: defaults.lens, motion: defaults.motion, zoom: defaults.zoom, respect_mask_edges: defaults.respect_mask_edges });
  const hasMask = item.masks.items.length > 0;
  const controls = { item, send };

  return (
    <>
      <Card title="Blur" action={changed(item) ? <ResetLink size="card" aria-label="Reset Blur" title={BLUR_TEXT.cardReset} onClick={reset} /> : undefined}>
        <div className="ym-blur__type">
          <Segmented options={options} value={item.type} onChange={(type) => send({ type })} label={BLUR_TEXT.typeLabel} />
        </div>
        {item.type === "soft" ? <SoftControls {...controls} /> : null}
        {item.type === "lens" ? <LensControls {...controls} /> : null}
        {item.type === "motion" ? <MotionControls {...controls} /> : null}
        {item.type === "zoom" ? <ZoomControls {...controls} picking={pickingCentre} onPick={onPickCentre ?? (() => undefined)} /> : null}
      </Card>
      <Card title={BLUR_TEXT.edgesTitle} last>
        <div className="ym-blur__edges">
          <div className="ym-blur__edges-text">
            <strong className="ym-blur__edges-name" aria-hidden="true">
              {BLUR_TEXT.edgesLabel}
            </strong>
            <span className="ym-blur__line">{item.respect_mask_edges ? BLUR_TEXT.edgesOn : BLUR_TEXT.edgesOff}</span>
            {hasMask ? null : <span className="ym-blur__line">{BLUR_TEXT.edgesNoMask}</span>}
          </div>
          <Switch label={BLUR_TEXT.edgesLabel} checked={item.respect_mask_edges} onChange={(on) => send({ respect_mask_edges: on })} />
        </div>
      </Card>
    </>
  );
}

/** The node's item from the resolved settings, or null when it is not a Blur node. */
export function blurItemOf(rs: ResolvedSettings, node: number): BlurItem | null {
  const item = nodeOf(rs.settings, node);
  return item?.kind === "blur" ? item : null;
}
