import { PipetteIcon } from "lucide-react";
import { Badge } from "@/components/glass/badge";
import { Spinner } from "@/components/glass/spinner";
import { useAppState } from "../state/context";

/** What the hint says while Pick depth is on (`docs/23` §7.6). */
export const PICK_DEPTH_HINT = "Click the part of the photo to select. Esc stops.";
/** What it says while the depth map is still being made: a click waits for it. */
export const PICK_DEPTH_WAIT = "Getting ready…";

/**
 * The hint of Pick depth (`docs/11` *The Depth card*): Pick white's pill at the top centre of the photo, with an eyedropper. While the core is making the photo's depth
 * map (`viewport_busy.finding`) it is a spinner and "Getting ready…" instead, because a click waits for the map. A polite status; it takes no pointer events.
 */
export function PickDepthHint() {
  const waiting = useAppState((s) => s.viewportBusy.busy && s.viewportBusy.finding);
  return (
    <Badge variant="outline" className="ym-photo-cell__hint" role="status" aria-live="polite" data-pick-depth-hint={waiting ? "waiting" : "ready"}>
      {waiting ? <Spinner aria-hidden="true" /> : <PipetteIcon aria-hidden="true" />}
      {waiting ? PICK_DEPTH_WAIT : PICK_DEPTH_HINT}
    </Badge>
  );
}
