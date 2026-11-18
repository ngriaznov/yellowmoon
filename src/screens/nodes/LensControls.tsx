import { Segmented, Slider } from "../../components";
import type { SegmentedOption } from "../../components";
import { BLUR_TEXT, Tip } from "./blurParts";
import type { BlurControlProps } from "./blurParts";

/** Round (0 blades) and 5 to 9 blades (`docs/33` §2.2). */
const SHAPES: ReadonlyArray<SegmentedOption<string>> = [
  { value: "0", label: "Round", ariaLabel: "Round" },
  ...[5, 6, 7, 8, 9].map((n) => ({ value: String(n), label: String(n), ariaLabel: `${n} blades` })),
];

/**
 * The Lens type's controls: Radius, Shape, Rotation and Highlights (`docs/33` §2.2). Rotation is off with Round, since a disc is its own
 * rotation; its value is kept.
 */
export function LensControls({ item, send }: BlurControlProps) {
  const lens = item.lens;
  const round = lens.blades === 0;
  return (
    <>
      <Tip text={BLUR_TEXT.radiusTip}>
        <Slider label="Radius" value={lens.radius} min={0} max={100} step={1} defaultValue={20} onChange={(v, final) => send({ lens: { radius: v } }, final)} />
      </Tip>
      <div className="ym-blur__shape">
        <span className="ym-blur__shape-label" id="ym-blur-shape">
          Shape
        </span>
        <Tip text={BLUR_TEXT.shapeTip}>
          <Segmented options={SHAPES} value={String(lens.blades)} onChange={(v) => send({ lens: { blades: Number(v) } })} label="Shape" size="sm" />
        </Tip>
      </div>
      <Slider
        label="Rotation"
        value={Math.round(lens.rotation_deg) % 360}
        min={0}
        max={359}
        step={1}
        defaultValue={0}
        format={(v) => `${Math.round(v)}°`}
        disabled={round}
        onChange={(v, final) => send({ lens: { rotation_deg: v } }, final)}
      />
      <Tip text={BLUR_TEXT.highlightsTip}>
        <Slider label="Highlights" value={lens.highlights} min={0} max={100} step={1} defaultValue={0} onChange={(v, final) => send({ lens: { highlights: v } }, final)} />
      </Tip>
      <p className="ym-blur__line">{BLUR_TEXT.lensLine}</p>
    </>
  );
}
