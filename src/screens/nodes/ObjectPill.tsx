import { SparklesIcon } from "lucide-react";
import { Badge } from "@/components/glass/badge";
import { useAppState } from "../../state/context";

/**
 * "Object found · 0.4 s" with a sparkle, at the top centre of the photo (`docs/23` §4.4, mockup 07): the time the decoder took, to one decimal, for three
 * seconds after a click or a box found a mask (`Store.findObject`). A polite status in the slot of the other notes; it takes no pointer events.
 */
export function ObjectPill() {
  const pill = useAppState((s) => s.objectPill);
  if (!pill) return null;
  return (
    <Badge key={pill.seq} variant="outline" className="ym-photo-cell__hint ym-photo-cell__found" role="status" aria-live="polite" data-object-pill="">
      <SparklesIcon aria-hidden="true" />
      {pill.text}
    </Badge>
  );
}
