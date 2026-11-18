import type { ReactNode } from "react";
import { TooltipTrigger } from "@/components/glass/tooltip";
import { OptionalTooltip } from "@/components/OptionalTooltip";
import type { BlurNode, SettingsPatch } from "../../api/types";

/** The copy of the Blur tab (`docs/33` §2; `design/blur/states.jpg`). British spelling, short plain sentences. */
export const BLUR_TEXT = {
  typeLabel: "Blur type",
  cardReset: "Sets the type and its settings back to the defaults. Masks stay.",
  radiusTip: "How far the blur reaches, as a share of the picture's size.",
  shapeTip: "A round aperture. Choose a number of blades for a polygon.",
  highlightsTip: "Makes bright lights count for more, so they form clear discs.",
  lengthTip: "How far the blur runs, as a share of the picture's size.",
  angleTip: "The direction of the blur. 0° is along the picture's width.",
  amountTip: "How much blur grows toward the edges.",
  pickTip: "Click the photo to set the centre. Esc stops.",
  softLine: "Radius is a share of the picture's size, so the view and every export look alike.",
  lensLine: "Bright lights become discs, or the shape of the blades. Highlights makes them stronger.",
  motionLine: "Length is a share of the picture's size. Angle 0° is along the picture's width.",
  zoomLine: "Pixels farther from the centre blur more. Amount 0 changes nothing.",
  centreAtMiddle: "Centre of the picture.",
  centreMoved: "Moved from the centre.",
  centreResetTip: "Puts the centre back in the middle of the picture.",
  edgesTitle: "Mask edges",
  edgesLabel: "Respect mask edges",
  edgesOn: "Colours from outside the mask are not blurred in, so there is no glow around the subject.",
  edgesOff: "The whole picture is blurred, then the mask decides where the blur shows.",
  edgesNoMask: "This node has no mask, so this changes nothing yet.",
} as const;

/**
 * Which types have a kernel that renders them. A type whose kernel has not landed is disabled in the Type control (`docs/15` BL-13).
 * All four are on: the node holds every type's values, the CPU oracle exists for each (BL-03) and BL-04 to BL-09 wire the runs; set a
 * type to `false` to grey it out (or to cut it) without touching anything else (`docs/33` §11 item 24).
 */
export const BLUR_TYPE_READY: Readonly<Record<BlurNode["type"], boolean>> = { soft: true, lens: true, motion: true, zoom: true };

/** What a type's controls get: the node, and a writer of one patch of the node (a drag sends one per frame; they fold into one undo step). */
export interface BlurControlProps {
  item: BlurNode;
  /** Writes a partial of the node as one patch. `final` is false while a slider is dragged. */
  send: (partial: Parameters<typeof blurPatch>[0], final?: boolean) => void;
}

/** A node patch body for the Blur node, typed to the fields a patch may set. */
export const blurPatch = (partial: NonNullable<SettingsPatch["nodes"]>[string]): NonNullable<SettingsPatch["nodes"]>[string] => partial;

/** A control with its one-line tooltip. */
export function Tip({ text, children }: { text: string; children: ReactNode }) {
  return (
    <OptionalTooltip text={text} side="top">
      <TooltipTrigger asChild>
        <div className="ym-blur-tip">{children}</div>
      </TooltipTrigger>
    </OptionalTooltip>
  );
}
