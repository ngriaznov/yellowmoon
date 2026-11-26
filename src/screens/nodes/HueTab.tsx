import { useRef, useState } from "react";
import type { ResolvedSettings } from "../../api/types";
import { Button, Card, ResetLink, RowNote } from "../../components";
import { FULL_SCALE, isFullScale, MAX_HUE_POINTS, widenedScale, zoomedScale } from "../../components/hueCurve";
import type { HueCurveKind, HuePoint, HueScale } from "../../components/hueCurve";
import { HuePlot } from "../../components/HuePlot";
import { patchOf } from "../../state/settingsGraph";
import type { ReferenceGradeItem } from "../../state/settingsGraph";
import type { AppStore } from "../../state/store";

/** What the Hue tab says (`docs/34` §9.4). */
export const HUE_TEXT = {
  full: "A curve can have up to 16 points.",
  fullRange: "Full range",
  fullRangeTip: "Show the whole range of the plot, or the part the points are in.",
} as const;

const CARDS: ReadonlyArray<{ kind: HueCurveKind; title: string; label: string }> = [
  { kind: "hue_vs_hue", title: "Hue vs hue", label: "Hue vs hue curve" },
  { kind: "hue_vs_sat", title: "Hue vs saturation", label: "Hue vs saturation curve" },
];

interface HueCardProps {
  store: AppStore;
  rs: ResolvedSettings;
  node: number;
  kind: HueCurveKind;
  title: string;
  label: string;
  points: readonly HuePoint[];
  last?: boolean;
}

/**
 * One hue curve's card: the plot (`HuePlot`) on a **zoomed default view that fits the fitted values**, with the full range one press away (the owner's
 * answer at RG-H2, `docs/34` §9.4), the readout of the selected point and the card's own Reset. The view opens on the points the curve has when the tab
 * opens, and only widens as points are moved past its edge; "Full range" shows ±180° (Hue vs hue) or ⅛ to 8 (Hue vs saturation).
 */
function HueCard({ store, rs, node, kind, title, label, points, last }: HueCardProps) {
  const [full, setFull] = useState(false);
  const [atMost, setAtMost] = useState(false);
  const opened = useRef<HueScale>(zoomedScale(kind, points));
  opened.current = widenedScale(opened.current, points);
  const scale = full ? FULL_SCALE[kind] : opened.current;
  const zoomable = !isFullScale(opened.current);
  const send = (list: HuePoint[], final: boolean): void => {
    setAtMost(false);
    void store.update(patchOf(node, { hue_curves: { [kind]: list } }), final, rs.photo_id);
  };
  return (
    <Card
      title={title}
      last={last}
      action={
        <>
          {zoomable ? (
            <Button variant="header" size="sm" pressed={full} title={HUE_TEXT.fullRangeTip} onClick={() => setFull(!full)}>
              {HUE_TEXT.fullRange}
            </Button>
          ) : null}
          <ResetLink size="card" aria-label={`Reset ${title}`} onClick={() => send([], true)} />
        </>
      }
    >
      <HuePlot kind={kind} points={points} scale={scale} label={label} onChange={send} onFull={() => setAtMost(points.length >= MAX_HUE_POINTS)} />
      {atMost ? <RowNote role="status">{HUE_TEXT.full}</RowNote> : null}
    </Card>
  );
}

/**
 * A Reference Grade node's Hue tab (`docs/34` §9.4): Hue vs hue and Hue vs saturation, two cards on the Curve card's plot widget with a periodic x axis. The
 * plots show the points the node holds; every change is one patch of the one curve (`hue_curves`), which is one undo step per gesture.
 */
export function HueTab({ store, rs, node, item }: { store: AppStore; rs: ResolvedSettings; node: number; item: ReferenceGradeItem }) {
  return (
    <>
      {CARDS.map(({ kind, title, label }, i) => (
        <HueCard key={kind} store={store} rs={rs} node={node} kind={kind} title={title} label={label} points={item.hue_curves[kind]} last={i === CARDS.length - 1} />
      ))}
    </>
  );
}
