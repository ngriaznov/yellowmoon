import { PipetteIcon } from "lucide-react";
import { Badge } from "@/components/glass/badge";

/** What the hint says while Pick white is on. */
export const PICK_WHITE_HINT = "Click something neutral: grey or white. Esc cancels.";

/**
 * The hint of Pick white (`docs/11` §Pick white): a glass pill with an eyedropper at the top centre of the photo, in the style of the
 * "Rendering on the CPU" pill (`RenderingNote`, whose slot it shares). A polite status; it takes no pointer events, so the crosshair
 * layer under it gets every click. When the CPU note shows too, the note moves down one pill (`.ym-photo-cell__hint ~ .ym-photo-cell__busy`).
 */
export function PickWhiteHint() {
  return (
    <Badge variant="outline" className="ym-photo-cell__hint" role="status" aria-live="polite">
      <PipetteIcon aria-hidden="true" />
      {PICK_WHITE_HINT}
    </Badge>
  );
}
