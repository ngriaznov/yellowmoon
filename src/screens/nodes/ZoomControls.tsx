import { CrosshairIcon } from "lucide-react";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import { Button, ResetLink, Slider } from "../../components";
import { BLUR_TEXT, Tip } from "./blurParts";
import type { BlurControlProps } from "./blurParts";

/** Whether the centre is off the middle of the picture. */
const moved = (c: readonly [number, number]): boolean => c[0] !== 0.5 || c[1] !== 0.5;

/**
 * The Zoom type's controls: Amount, and the Centre readout with its Reset (`docs/33` §2.2). Reset is one patch of `zoom.center`, one undo
 * step. Pick centre on the photo, and the handle, are BL-14's: the button is drawn and off until then.
 */
export function ZoomControls({ item, send }: BlurControlProps) {
  const away = moved(item.zoom.center);
  return (
    <>
      <Tip text={BLUR_TEXT.amountTip}>
        <Slider label="Amount" value={item.zoom.amount} min={0} max={100} step={1} defaultValue={20} onChange={(v, final) => send({ zoom: { amount: v } }, final)} />
      </Tip>
      <div className="ym-blur__centre" role="group" aria-label="Centre">
        {/* TODO(BL-14): Pick centre is a toggle (`pressed`) that turns the pick on over the photo; until then it is off. */}
        <OptionalTooltip text={BLUR_TEXT.pickTip} side="top">
          <TooltipTrigger asChild>
            <span className="ym-blur-tip">
              <Button variant="header" size="sm" disabled icon={<CrosshairIcon aria-hidden="true" />}>
                Pick centre
              </Button>
            </span>
          </TooltipTrigger>
        </OptionalTooltip>
        <span className="ym-blur__line ym-blur__centre-line">{away ? BLUR_TEXT.centreMoved : BLUR_TEXT.centreAtMiddle}</span>
        {away ? <ResetLink aria-label="Reset centre" title={BLUR_TEXT.centreResetTip} onClick={() => send({ zoom: { center: [0.5, 0.5] } })} /> : null}
      </div>
      <p className="ym-blur__line">{BLUR_TEXT.zoomLine}</p>
    </>
  );
}
