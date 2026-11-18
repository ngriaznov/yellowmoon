import { Slider } from "../../components";
import { BLUR_TEXT, Tip } from "./blurParts";
import type { BlurControlProps } from "./blurParts";

/** The Motion type's controls: Length (0 to 100, default 20) and Angle (0 to 179°, default 0) (`docs/33` §2.2). */
export function MotionControls({ item, send }: BlurControlProps) {
  return (
    <>
      <Tip text={BLUR_TEXT.lengthTip}>
        <Slider label="Length" value={item.motion.length} min={0} max={100} step={1} defaultValue={20} onChange={(v, final) => send({ motion: { length: v } }, final)} />
      </Tip>
      <Tip text={BLUR_TEXT.angleTip}>
        <Slider
          label="Angle"
          value={Math.round(item.motion.angle_deg) % 180}
          min={0}
          max={179}
          step={1}
          defaultValue={0}
          format={(v) => `${Math.round(v)}°`}
          onChange={(v, final) => send({ motion: { angle_deg: v } }, final)}
        />
      </Tip>
      <p className="ym-blur__line">{BLUR_TEXT.motionLine}</p>
    </>
  );
}
