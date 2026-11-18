import { Slider } from "../../components";
import { BLUR_TEXT, Tip } from "./blurParts";
import type { BlurControlProps } from "./blurParts";

/** The default Radius of a Soft blur (`docs/33` §2.5). */
const RADIUS_DEFAULT = 20;

/** The Soft type's controls: Radius, and the line that names the type's trait (`docs/33` §2.2). */
export function SoftControls({ item, send }: BlurControlProps) {
  return (
    <>
      <Tip text={BLUR_TEXT.radiusTip}>
        <Slider label="Radius" value={item.soft.radius} min={0} max={100} step={1} defaultValue={RADIUS_DEFAULT} onChange={(v, final) => send({ soft: { radius: v } }, final)} />
      </Tip>
      <p className="ym-blur__line">{BLUR_TEXT.softLine}</p>
    </>
  );
}
