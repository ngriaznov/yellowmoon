import { memo, useEffect, useMemo, useState } from "react";
import { Card, centralInnerPoint, CurvePlot, CURVE_CHANNEL_LABELS, ResetLink, Segmented } from "../../components";
import type { CurveChannel } from "../../components";
import type { Curves } from "../../api/types";
import { useAppState } from "../../state/context";
import type { TabProps } from "./shared";

const CHANNELS = Object.keys(CURVE_CHANNEL_LABELS) as CurveChannel[];

const IDENTITY: Curves = {
  luma: [[0, 0], [1, 1]],
  red: [[0, 0], [1, 1]],
  green: [[0, 0], [1, 1]],
  blue: [[0, 0], [1, 1]],
};

/** The histogram behind the plot: the channel's own bins, or the luminance of the three for Luma. */
function plotHistogram(channel: CurveChannel, data: { red: number[]; green: number[]; blue: number[] } | null): number[] | null {
  if (!data) return null;
  if (channel !== "luma") return data[channel];
  return data.green.map((g, i) => 0.2126 * (data.red[i] ?? 0) + 0.7152 * g + 0.0722 * (data.blue[i] ?? 0));
}

/** Curve: Luma, Red, Green and Blue curves on a 320 × 220 plot with the photo's histogram behind it. */
export const CurveTab = memo(function CurveTab({ store, photo, rs }: TabProps) {
  const [channel, setChannel] = useState<CurveChannel>("luma");
  const histogram = useAppState((s) => s.histogram);
  // The histogram behind the plot follows the frames the viewport draws, at most five times a second (`store.watchHistogram`).
  useEffect(() => store.watchHistogram(), [store, photo.id]);
  const background = useMemo(() => plotHistogram(channel, histogram), [channel, histogram]);
  const curve = rs.settings.curve;

  return (
    <Card title="Curve" action={<ResetLink size="card" onClick={() => void store.update({ curve: IDENTITY })} />}>
      <div className="ym-stack">
        <Segmented
          label="Curve channel"
          value={channel}
          onChange={setChannel}
          options={CHANNELS.map((c) => ({ value: c, label: CURVE_CHANNEL_LABELS[c] }))}
        />
        <CurvePlot
          key={channel}
          channel={channel}
          points={curve[channel]}
          histogram={background}
          defaultSelected={centralInnerPoint(curve[channel])}
          onChange={(points, final) => void store.update({ curve: { ...curve, [channel]: points } }, final, rs.photo_id)}
        />
      </div>
    </Card>
  );
});
