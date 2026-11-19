import { CrosshairIcon } from "lucide-react";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import { Button, ResetLink, Slider } from "../../components";
import { BLUR_TEXT, Tip } from "./blurParts";
import type { BlurControlProps } from "./blurParts";

/** What the Zoom controls get besides the node: whether Pick centre is on, and how to turn it on or off. */
export interface ZoomControlsProps extends BlurControlProps {
  picking: boolean;
  onPick: (on: boolean) => void;
}

/** Whether the centre is off the middle of the picture. */
const moved = (c: readonly [number, number]): boolean => c[0] !== 0.5 || c[1] !== 0.5;

/**
 * The Zoom type's controls: Amount, and the Centre with Pick centre, its line and its Reset (`docs/33` §2.2). Pick centre is a toggle that turns the
 * pick on over the photo (`Photos` keeps it, as the other picks; Esc ends it); the click, a drag of the handle and Reset are each one patch of
 * `zoom.center`, one undo step.
 */
export function ZoomControls({ item, send, picking, onPick }: ZoomControlsProps) {
  const away = moved(item.zoom.center);
  return (
    <>
      <Tip text={BLUR_TEXT.amountTip}>
        <Slider label="Amount" value={item.zoom.amount} min={0} max={100} step={1} defaultValue={20} onChange={(v, final) => send({ zoom: { amount: v } }, final)} />
      </Tip>
      <div className="ym-blur__centre" role="group" aria-label="Centre">
        <OptionalTooltip text={BLUR_TEXT.pickTip} side="top">
          <TooltipTrigger asChild>
            <span className="ym-blur-tip">
              <Button variant="header" size="sm" pressed={picking} icon={<CrosshairIcon aria-hidden="true" />} onClick={() => onPick(!picking)}>
                Pick centre
              </Button>
            </span>
          </TooltipTrigger>
        </OptionalTooltip>
        <span className="ym-blur__line ym-blur__centre-line">{picking ? BLUR_TEXT.centrePicking : away ? BLUR_TEXT.centreMoved : BLUR_TEXT.centreAtMiddle}</span>
        {away ? <ResetLink aria-label="Reset centre" title={BLUR_TEXT.centreResetTip} onClick={() => send({ zoom: { center: [0.5, 0.5] } })} /> : null}
      </div>
      <p className="ym-blur__line">{BLUR_TEXT.zoomLine}</p>
    </>
  );
}
