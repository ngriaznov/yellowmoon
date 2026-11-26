import { ImagesIcon } from "lucide-react";
import { Badge } from "@/components/glass/badge";
import { useAppState } from "../state/context";
import { REFERENCE_TEXT } from "../state/reference";

/** What the hint says while the strip is a chooser for a Reference Grade node's reference (`docs/34` §9.2). */
export const CHOOSER_HINT = REFERENCE_TEXT.chooserHint;

/**
 * The hint of the strip chooser: the glass pill of Pick white's hint (`PickWhiteHint`) with a picture glyph, at the top centre of the photo, a polite status. It
 * shows while a Reference Grade node's chooser is on (`store.choosing`); the screen ends the chooser when its tab, node or view goes, so the hint goes with it.
 */
export function PickReferenceHint() {
  const on = useAppState((s) => s.choosing !== null);
  if (!on) return null;
  return (
    <Badge variant="outline" className="ym-photo-cell__hint" role="status" aria-live="polite">
      <ImagesIcon aria-hidden="true" />
      {CHOOSER_HINT}
    </Badge>
  );
}
