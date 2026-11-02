import { useReactFlow, useStore } from "@xyflow/react";
import { MinusIcon, PlusIcon } from "lucide-react";
import { Button } from "../../components";

/** The zoom range of the canvas: 25 % to 200 % (`docs/23` §7.2). */
export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 2;
/** What "+" and "−" multiply the zoom by. */
export const ZOOM_STEP = 1.25;
/** What Fit shows: the whole graph with this much margin round it, never magnified beyond 100 %. */
export const FIT_OPTIONS = { padding: 0.12, minZoom: MIN_ZOOM, maxZoom: 1, duration: 160 } as const;

/** The zoom a press of "+" (1) or "−" (−1) goes to, kept in the range. */
export function steppedZoom(zoom: number, direction: 1 | -1): number {
  const next = direction === 1 ? zoom * ZOOM_STEP : zoom / ZOOM_STEP;
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(next * 1000) / 1000));
}

/**
 * The zoom control at the canvas's bottom right (`docs/23` §7.2): "−", the zoom as a percentage, "+" and "Fit" (names "Zoom out", "Zoom in"
 * and "Fit the graph"). A pinch or ⌘-scroll zooms too, and a two-finger scroll pans (`NodeGraph`).
 */
export function ZoomControl() {
  const flow = useReactFlow();
  const zoom = useStore((s) => s.transform[2]);
  const percent = Math.round(zoom * 100);
  return (
    <div className="ym-zoom nodrag nopan" role="group" aria-label="Zoom">
      <Button variant="secondary" size="sm" aria-label="Zoom out" disabled={zoom <= MIN_ZOOM + 1e-6} icon={<MinusIcon aria-hidden="true" />} onClick={() => void flow.zoomTo(steppedZoom(flow.getZoom(), -1))} />
      <span className="ym-zoom__value">{percent}%</span>
      <Button variant="secondary" size="sm" aria-label="Zoom in" disabled={zoom >= MAX_ZOOM - 1e-6} icon={<PlusIcon aria-hidden="true" />} onClick={() => void flow.zoomTo(steppedZoom(flow.getZoom(), 1))} />
      <Button variant="secondary" size="sm" aria-label="Fit the graph" onClick={() => void flow.fitView(FIT_OPTIONS)}>
        Fit
      </Button>
    </div>
  );
}
